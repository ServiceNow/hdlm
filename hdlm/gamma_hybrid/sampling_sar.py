import abc
import torch
from hdlm.gamma_hybrid.catsample import sample_categorical

import hdlm.model.utils as mautils
import hdlm.metaschedule_utils as msutils

import torch
from time import time


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, positions, t, step_size, attn_mask):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, positions, t, step_size, attn_mask, tokens_to_denoise):
        t = t.expand(*x.shape)
        sigma, dsigma = self.noise(t)
        log_score = score_fn(x, sigma, positions=positions, attn_mask=attn_mask)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, log_score.exp()) 
        if self.graph.absorb:
            rev_rate[..., -1] = torch.where(
                    tokens_to_denoise.squeeze(-1),
                    torch.zeros_like(rev_rate[..., -1]),
                    rev_rate[..., -1]
                )
        return self.graph.sample_rate(x, rev_rate)

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, positions, t, step_size, attn_mask, tokens_to_denoise):
        return x

# Tweedie Denoising
@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, positions, t, step_size, attn_mask, tokens_to_denoise, active_mask):
        t = t.expand(*x.shape)
        curr_sigma = self.noise(t)[0] # σ[bar](t)
        next_sigma = self.noise(t - step_size)[0] # σ[bar](t - ∆t)
        dsigma = curr_sigma - next_sigma #(σ[bar](t) - σ[bar](t - ∆t))
        log_score = score_fn(x, curr_sigma, positions=positions, attn_mask=attn_mask)
        stag_score = self.graph.staggered_score(log_score.exp(), dsigma) # exp(σ[bar](t) - σ[bar](t - ∆t))Q)*score
        probs = stag_score * self.graph.transp_transition(x, dsigma) # exp(σ[bar](t) - σ[bar](t - ∆t))Q)*score * exp(σ[bar](t) - σ[bar](t - ∆t))Q^Tt)
        if self.graph.absorb:
            probs[..., -1] = torch.where(
                    tokens_to_denoise,
                    torch.zeros_like(probs[..., -1]),
                    probs[..., -1]
                )
        x_updated = sample_categorical(probs)
        return torch.where(active_mask, x_updated, x)

# sa for simple annealing
def get_sa_sampling_fn(config, graph, noise, meta_schedule, batch_dims, eps, device, proj_fun=lambda x: x):
    
    sampling_fn = get_sa_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 metaschedule=meta_schedule,
                                 annealing=config.annealing,
                                 eps=eps,
                                 device=device,
                                 proj_fun=proj_fun,
                                 )
    
    return sampling_fn



def get_sa_sampler(graph, noise, batch_dims, predictor, metaschedule, annealing, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    active_eps = 10 * eps
    step_size = (1 - 2 * active_eps) / (metaschedule.worthless - 1)
    @torch.no_grad()
    def pc_sampler(model):
        start = time()
        sampling_score_fn = mautils.get_score_fn(model, train=False, sampling=True)
        batch_size, seq_len = batch_dims
        full_sample = graph.sample_limit(batch_size, seq_len).to(device)
        x = None
        active_window_len = 0
        for current_step in range(1, metaschedule.max_step(seq_len)):
            num_settled = metaschedule(current_step).num_settled
            new_active_window_len = min(metaschedule(current_step).relevant.__len__() + num_settled, seq_len)
            t, attn_mask, tokens_to_denoise, active_mask, settled_mask, _ = msutils.compute_t_and_attn(
                batch_size, new_active_window_len, metaschedule, current_step, annealing.attention.context_type, annealing.attention.block_type, device, eps)
            if new_active_window_len > active_window_len:
                new_x = full_sample[:, active_window_len: new_active_window_len]
                if x is None:
                    x = new_x
                else:
                    x = torch.concat([x, new_x], dim=1)

            # if new_active_window_len > active_window_len:
            #     print(f"step {current_step}, len {new_active_window_len}, time {time() - start}")

            active_window_len = new_active_window_len
            step = torch.ones_like(t) * step_size
            step[tokens_to_denoise] = 0.9 * active_eps
            step[settled_mask] = 0.9 * eps
            t = t.unsqueeze(0).expand(batch_size, -1)
            step = step.unsqueeze(0).expand(batch_size, -1)
            step = torch.clamp(step, min=0.9 * eps, max=step_size)

            tokens_to_denoise = tokens_to_denoise.unsqueeze(0).expand(batch_size, -1)
            active_mask = active_mask.unsqueeze(0).expand(batch_size, -1)
            positions = torch.arange(x.shape[1], device=device).repeat(batch_size, 1)
            x = projector(x)                
            x = predictor.update_fn(sampling_score_fn, x, positions, t, step, attn_mask, tokens_to_denoise, active_mask)
            
        print(f"Sampling time: {time() - start} for {metaschedule.max_step(seq_len)} steps and {x.shape[0]} samples.")
            
        return x

    return pc_sampler