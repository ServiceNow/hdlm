import abc
from typing import Union
import torch
import torch.nn.functional as F


from hdlm.gamma_hybrid.catsample import sample_categorical

def get_graph(config, device):
    try:
        tokens = config.tokenizer.tokens
    except:
        tokens = config.tokens
    expanded = config.graph.expanded_sigma if hasattr(config.graph, "expanded_sigma") else False
    if config.graph.type == "uniform":
        return Uniform(tokens, expanded=expanded)
    elif config.graph.type == "absorb":
        return Absorbing(tokens, expanded=expanded)
    elif config.graph.type == "QGamma":
        return QGamma(tokens, config.graph.gamma, expanded=expanded)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass

    @property
    @abc.abstractmethod
    def Q(self):
        """
        Computes the full rate matrix Q.

        Main usage: tests cases (including checking the maths).
        """
    
    def full_transition(self, sigma):
        """
        Computes the full (forward) transition matrix.
        
        This default implementation relies on `matrix_exp`.
        """
        return torch.matrix_exp(sigma * self.Q)

    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass

    @abc.abstractmethod
    def transp_transition(self, i, sigma):
        """
        Compute the i-th row of the transition matrix e^{sigma Q}.
        """
        pass

    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)
    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    
    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class ScaleIdempotentGraph(Graph):
    """
    Graph for which squaring the rate matrix Q amounts to scaling it.

    Q's eigenvalues may thus only be zero and/or that scale.
    """
    @property
    def idempotence_scale(self) -> Union[int, float]:
        """
        self.Q @ self.Q == self.idempotence_scale * self.Q
        """
        pass

    def full_transition(self, sigma):
        """
        Computes the full (forward) transition matrix.
        
        This implementation leverages the scale idempotence property.
        """
        factor = (torch.tensor(self.idempotence_scale * sigma).exp()-1)/self.idempotence_scale
        return torch.eye(self.dim) + factor * self.Q


class AntiIdempotentGraph(ScaleIdempotentGraph):
    @property
    def idempotence_scale(self) -> int:
        """
        Q @ Q == -Q
        """
        return -1


def Q_uniform(dim, *, last_dim_is_absorbing = False):
    N = dim - 1 if last_dim_is_absorbing else dim
    out = (torch.ones((dim, dim)) - N*torch.eye(dim))/N
    if last_dim_is_absorbing:
        out[:,-1] = 0
        out[-1, :] = 0
    return out


def Q_absorb(dim):
    out = torch.zeros((dim, dim), dtype=torch.int64)
    out[-1, :] = 1
    out -= torch.eye(dim, dtype=torch.int64)
    return out


class Uniform(AntiIdempotentGraph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim, expanded=False):
        self._dim = dim
        self.expanded_sigma = expanded

    @property
    def dim(self):
        return self._dim
    
    @property
    def absorb(self):
        return False
    
    @property
    def Q(self):
        return Q_uniform(self.dim)

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)
    
    def transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)


    def sample_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i)
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        dsigma = unsqueeze_as(dsigma, score)
        dim = score.shape[-1]
        epow = (-dsigma).exp()
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)
        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )
        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )
        
        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        print('pos_term(Uniform):', pos_term)
        print('neg_term(Uniform):', neg_term)
        print('const(Uniform):', const)
        return pos_term - neg_term + const


class Absorbing(AntiIdempotentGraph):
    def __init__(self, dim, expanded=False):
        super().__init__()
        self._dim = dim
        self.expanded_sigma = expanded

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    @property
    def Q(self):
        return Q_absorb(self.dim)

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i)
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert


    def staggered_score(self, score, dsigma):
        dsigma = unsqueeze_as(dsigma, score)
        score = score.clone()
        extra_const = (1 - dsigma.exp().squeeze(-1)) * score.sum(dim=-1)
        score *= dsigma.exp()
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)
    
    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1  # Shape: [batch_size, seq_len]

        # Adjust sigma to match the shape of x
        if sigma.dim() == 2 and sigma.size(1) == 1:
            # sigma is [batch_size, 1], expand to [batch_size, seq_len]
            sigma = sigma.expand(-1, x.size(1))
        elif sigma.dim() == 2 and sigma.size(1) == x.size(1):
            # sigma is already [batch_size, seq_len]
            pass
        else:
            raise ValueError(f"Unexpected shape of sigma: {sigma.shape}")

        # Compute esigm1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )  # Shape: [batch_size, seq_len]

        # Gather esigm1 values where x == absorbing state
        esigm1_rel = esigm1[rel_ind]  # Shape: [num_absorbing]

        ratio = 1 / esigm1_rel  # Shape: [num_absorbing]
        other_ind = x0[rel_ind]  # Shape: [num_absorbing]

        # Negative term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        # Positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # Constant term
        const = ratio * (ratio.log() - 1)

        # Initialize entropy tensor
        entropy = torch.zeros_like(x, dtype=score.dtype)
        entropy[rel_ind] = pos_term - neg_term + const
        return entropy

class QGamma(Graph):
    """
    A gamma transition matrix which combines absorbing and uniform transition matrices:
    Q_gamma = (1-gamma) * Q_absorb + gamma * Q_uniform
    """
    def __init__(self, dim, gamma, expanded=False):
        """
        Initialize Hybrid graph with dimension dim and mixing parameter gamma.
        """
        self._dim = dim
        self.gamma = gamma
        self.expanded_sigma = expanded

    # @done
    @property
    def dim(self):
        """
        Return the total number of states, including the absorbing state.
        """
        return self._dim + 1

    @property
    def absorb(self):
        """
        Return True since the last state (dim - 1) is an absorbing state.
        """
        return True

    # @done
    @property
    def Q(self):
        return (1-self.gamma) * Q_absorb(self.dim) + self.gamma * Q_uniform(self.dim, last_dim_is_absorbing=True)
    # @done
    def Q_a(self, i):
        """
        Computes the absorbing matrix Q_a where the last state is absorbing.
        """
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)
    
    # @done
    def Q_a_T(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge
    # @done
    def Q_u(self, i):
        """
        Computes the uniform matrix Q_u with absorbing state as the last row and column.
        Additionally, the last value in each row should be set to zero.
        """
        N = self.dim - 1  # The size of the original matrix (without absorbing state)
        
        # Initialize a (N+1) x (N+1) matrix
        edge = torch.ones(*i.shape, N+1, device=i.device) / N

        # Scatter to adjust the diagonal self-transition (set self-transitions correctly)
        edge = edge.scatter(-1, i[..., None], -(N - 1) / N)

        # Set the last value in each row to zero (last column)
        last_state = self.dim - 1
        edge[..., last_state] = 0  # Set the last column to 0 for every row
        
        # For rows where `i` equals the absorbing state, set the entire row to zero
        absorbing_mask = (i == last_state).unsqueeze(-1)  # Create mask where `i == 31999`
        edge = torch.where(absorbing_mask, torch.zeros_like(edge), edge)
        return edge
    
    # @done
    def rate(self, i):
        """
        Compute the forward rate matrix Q_hybrid = Q_absorb + gamma * Q_uniform.
        """
        return (1 - self.gamma) * self.Q_a(i) + self.gamma * self.Q_u(i)

    # @done
    def transp_rate(self, i):
        """
        Compute the transpose of the rate matrix (reverse process).
        """
        # Absorbing matrix transpose (same as the forward absorbing matrix)
        absorb_edge = self.Q_a_T(i)
        # Transpose of the uniform matrix
        uniform_edge = self.Q_u(i)
        return (1 - self.gamma) * absorb_edge + self.gamma * uniform_edge

    # @done
    def sample_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i)
        
        # A = e^{-σ}: probability of no absorbing jump.
        # B = e^{-α·σ}: given no absorbing jump, probability to not make a uniform move.
        A = torch.exp(-(1 - self.gamma) * sigma)
        B = torch.exp(-self.gamma * sigma)
        
        # For nonabsorbing states (i < self.dim - 1):
        P_absorb = 1 - A            # move to absorbing state.
        P_uniform = A * (1 - B)       # move to a different nonabsorbing state.
        # (P_no_move = A * B is implicit: if neither event occurs, we stay.)
        
        r = torch.rand(*i.shape, device=i.device)
        
        i_pert = i.clone()
        
        nonabs_mask = (i != self.dim - 1)
        
        absorb_mask = (r < P_absorb) & nonabs_mask
        uniform_mask = (r >= P_absorb) & (r < (P_absorb + P_uniform)) & nonabs_mask
        
        i_pert[absorb_mask] = self.dim - 1
        
        if uniform_mask.any():
            num_uniform = uniform_mask.sum().item()
            rand_states = torch.randint(0, self.dim - 1, (num_uniform,), device=i.device)
            i_pert[uniform_mask] = rand_states
        
        return i_pert

    def sample_limit(self, *batch_dims):
        """
        Samples the limiting distribution of the Qgamma model.
        """
        if self.gamma == 1:
            return torch.randint(0, self.dim - 1, batch_dims)
        sigma_max = torch.tensor(4.0)
        absorbing_prob = (1 - torch.exp(-sigma_max * (1 - self.gamma)))
        uniform_prob = (1 - torch.exp(-sigma_max * self.gamma)) * torch.exp(-sigma_max * (1 - self.gamma))
        total_prob = absorbing_prob + uniform_prob
        absorbing_rate = absorbing_prob / total_prob
        rand = torch.rand(*batch_dims)
        return torch.where(rand < absorbing_rate, torch.full(batch_dims, self.dim - 1), torch.randint(0, self.dim - 1, batch_dims))
        
    def transition(self, i, sigma):
        pass

    # @done
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        cumnoise = sigma
        exp_neg_cumnoise = torch.exp(-(1 - self.gamma) * cumnoise)
        exp_neg_gamma_cumnoise = torch.exp(-self.gamma * cumnoise)
        N = self.dim - 1  # Number of non-absorbing states
        
        batch_size, seq_len = i.shape
        trans = torch.zeros(batch_size, seq_len, self.dim, device=i.device)
        
        is_absorbing = (i == N)
        not_absorbing = ~is_absorbing
        
        # Handle absorbing states
        if is_absorbing.any():
            idx_absorb = is_absorbing.nonzero(as_tuple=False)
            batch_idx, seq_idx = idx_absorb[:, 0], idx_absorb[:, 1]
            
            # From absorbing state to non-absorbing states via absorbing component
            P_absorb_to_non_absorb = (1 - exp_neg_cumnoise[batch_idx, seq_idx, 0])
            trans[batch_idx, seq_idx, :N] = P_absorb_to_non_absorb.unsqueeze(-1).expand(-1, N)
                        
            # Staying in the absorbing state
            P_stay_absorb = 1.0
            trans[batch_idx, seq_idx, N] += P_stay_absorb
        
        # Handle non-absorbing states
        if not_absorbing.any():
            idx_non_absorb = not_absorbing.nonzero(as_tuple=False)
            batch_idx, seq_idx = idx_non_absorb[:, 0], idx_non_absorb[:, 1]
            current_states = i[batch_idx, seq_idx]
            
            # From other non-absorbing states via uniform component
            P_non_absorb_uniform = exp_neg_cumnoise[batch_idx, seq_idx, 0] * (1 - exp_neg_gamma_cumnoise[batch_idx, seq_idx, 0]) / N
            
            # Assign probabilities from other non-absorbing states
            trans[batch_idx, seq_idx, :N] += P_non_absorb_uniform.unsqueeze(-1).expand(-1, N)
            # Exclude self-transition
            # trans[batch_idx, seq_idx, current_states] -= P_non_absorb_uniform
            
            # Staying in the same state
            P_stay = exp_neg_cumnoise[batch_idx, seq_idx, 0] * exp_neg_gamma_cumnoise[batch_idx, seq_idx, 0]
            trans[batch_idx, seq_idx, current_states] += P_stay
                
        return trans


    def score_entropy(self, log_score, sigma, x, x0):
        r"""
        Compute the QGamma score-entropy loss for a discrete diffusion language model.
        
        Q_α = (1 - α) · Q_a + α · Q_u.
        
        - For tokens where x == mask (self.dim - 1), only the absorbing branch (Q_a)
        contributes.
        - For tokens where x is nonabsorbing (x < self.dim - 1), the loss is computed
        using the uniform branch. For this branch we only consider channels 0 .. effective_dim-1,
        where effective_dim = self.dim - 1. If x0 is not in that range, we set loss=0.
        
        Args:
        log_score: [bs, seq_len, vocab_dim] log-probability scores.
        sigma: [bs, seq_len] or [bs, 1] noise level.
        x: [bs, seq_len] current token indices.
        x0: [bs, seq_len] original token indices.
        
        Returns:
        Tensor of shape [bs, seq_len] containing the score-entropy loss.
        """
        bs, seq_len, vocab_dim = log_score.shape
        device = log_score.device
        epsilon = 1e-12

        # Ensure sigma has shape [bs, seq_len]
        if sigma.dim() == 2:
            if sigma.size(1) == 1:
                sigma = sigma.expand(bs, seq_len)
            elif sigma.size(1) != seq_len:
                raise ValueError(f"Unexpected sigma shape: {sigma.shape}")
        else:
            raise ValueError(f"Unexpected sigma dimensions: {sigma.shape}")

        # Compute esigm1
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)

        # Coefficients:
        ratio_absorb = 1 / (esigm1 + epsilon)
        effective_dim = self.dim - 1  # only nonabsorbing channels
        ratio_uniform = 1 - effective_dim / (esigm1 + effective_dim + epsilon)

        # Precompute exponential scores.
        exp_score = torch.exp(log_score)
        # For uniform branch, restrict to nonabsorbing channels.
        score_nonabs = log_score[..., :effective_dim]       # shape [bs, seq_len, effective_dim]
        exp_score_nonabs = torch.exp(score_nonabs)
        score_mean_nonabs = score_nonabs.mean(dim=-1)         # [bs, seq_len]
        exp_mean_nonabs = exp_score_nonabs.mean(dim=-1)         # [bs, seq_len]

        # Identify absorbing tokens (mask = self.dim - 1) vs. nonabsorbing.
        is_absorbing = (x == self.dim - 1)
        is_nonabsorbing = ~is_absorbing

        loss = torch.zeros(bs, seq_len, device=device, dtype=log_score.dtype)

        # CASE 1: Absorbing tokens: use Q_a branch.
        if is_absorbing.any():
            idx_abs = is_absorbing.nonzero(as_tuple=False)  # [n,2]
            bs_idx, seq_idx = idx_abs[:, 0].view(-1), idx_abs[:, 1].view(-1)
            x0_abs = x0[bs_idx, seq_idx]  # these indices can be in [0, self.dim)
            score_x0_abs = log_score[bs_idx, seq_idx, x0_abs]
            neg_term_abs = ratio_absorb[bs_idx, seq_idx] * score_x0_abs
            pos_term_abs = torch.exp(log_score[bs_idx, seq_idx, :effective_dim]).sum(dim=-1)
            const_abs = ratio_absorb[bs_idx, seq_idx] * (torch.log(ratio_absorb[bs_idx, seq_idx] + epsilon) - 1)
            loss[bs_idx, seq_idx] = (pos_term_abs - neg_term_abs + const_abs) * (1 - self.gamma)

        # CASE 2: Nonabsorbing tokens: use uniform branch.
        if is_nonabsorbing.any():
            idx_non = is_nonabsorbing.nonzero(as_tuple=False)  # [m,2]
            bs_idx, seq_idx = idx_non[:, 0].view(-1), idx_non[:, 1].view(-1)
            # For nonabsorbing tokens, x must be in [0, effective_dim).
            # x0 should also be in that range; otherwise, we set loss = 0.
            x_non = x[bs_idx, seq_idx].view(-1)     # shape [m]
            x0_non = x0[bs_idx, seq_idx].view(-1)     # shape [m]
            valid = (x0_non < effective_dim)
            # Initialize loss values for nonabsorbing tokens.
            loss_non = torch.zeros_like(x_non, dtype=log_score.dtype)
            if valid.any():
                valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
                bs_valid = bs_idx[valid_idx]
                seq_valid = seq_idx[valid_idx]
                x_non_valid = x[bs_valid, seq_valid].view(-1)      # current token indices
                x0_non_valid = x0[bs_valid, seq_valid].view(-1)      # original token indices; now guaranteed < effective_dim
                # Gather current token scores from score_nonabs.
                current_scores = score_nonabs[bs_valid, seq_valid]  # [m_valid, effective_dim]
                score_current = current_scores.gather(-1, x_non_valid.unsqueeze(-1)).squeeze(-1)
                exp_current = exp_score_nonabs[bs_valid, seq_valid].gather(-1, x_non_valid.unsqueeze(-1)).squeeze(-1)
                # Compute uniform common terms:
                uniform_neg_common = score_mean_nonabs[bs_valid, seq_valid] - score_current / effective_dim
                uniform_pos_common = exp_mean_nonabs[bs_valid, seq_valid] - exp_current / effective_dim

                # Split into two subcases: same (x == x0) and different.
                same_mask = (x_non_valid == x0_non_valid)
                neg_term = torch.zeros_like(uniform_neg_common)
                const_term = torch.zeros_like(uniform_neg_common)
                if same_mask.any():
                    idx_same = same_mask.nonzero(as_tuple=False).squeeze(-1)
                    neg_term[idx_same] = ratio_uniform[bs_valid[idx_same], seq_valid[idx_same]] * uniform_neg_common[idx_same]
                    const_term[idx_same] = ((effective_dim - 1) / effective_dim *
                                            ratio_uniform[bs_valid[idx_same], seq_valid[idx_same]] *
                                            (torch.log(ratio_uniform[bs_valid[idx_same], seq_valid[idx_same]] + epsilon) - 1))
                if (~same_mask).any():
                    idx_diff = (~same_mask).nonzero(as_tuple=False).squeeze(-1)
                    # For diff case, include extra term: score at x0 from nonabsorbing channels.
                    current_scores_diff = score_nonabs[bs_valid[idx_diff], seq_valid[idx_diff]]
                    score_x0_diff = current_scores_diff.gather(-1, x0_non_valid[idx_diff].unsqueeze(-1)).squeeze(-1)
                    neg_term[idx_diff] = uniform_neg_common[idx_diff] + score_x0_diff / (esigm1[bs_valid[idx_diff], seq_valid[idx_diff]] + epsilon)
                    const_term[idx_diff] = ((-torch.log(ratio_uniform[bs_valid[idx_diff], seq_valid[idx_diff]] + epsilon) - 1) /
                                                (ratio_uniform[bs_valid[idx_diff], seq_valid[idx_diff]] + epsilon) - (effective_dim - 2)) / effective_dim
                loss_non[valid_idx] = self.gamma * (uniform_pos_common - neg_term + const_term)
                # For invalid positions (where x0_non >= effective_dim), loss remains 0.
            # Write back loss for nonabsorbing tokens.
            loss[bs_idx, seq_idx] = loss_non

        return loss



    def staggered_score(self, score, dsigma):
        r"""
        Computes the staggered score for the Hybrid graph 
        Qₐₗpha = (1 - α)·Qₐ + α·Qᵤ.

        We first compute the transformation corresponding to the 
        absorbing branch (as in Q_absorb) and the transformation 
        corresponding to the uniform branch (as in Q_uniform), but for 
        the uniform branch we sum only over nonabsorbing tokens 
        (i.e. channels 0..dim-2) and normalize by (dim-1).

        Then we combine them:
        
            staggered_score = (1 - α) * score_abs + α * score_uni

        using the same dsigma for both branches.

        Args:
            score (torch.Tensor): The model’s estimated log-scores,
            shape [bs, seq_len, vocab_dim] (vocab_dim includes the absorbing token).
            dsigma (torch.Tensor): The noise increment, shape [bs, 1] or [bs, seq_len].
                
        Returns:
            torch.Tensor: The adjusted (staggered) score, shape [bs, seq_len, vocab_dim].
        """
        # Ensure dsigma has shape [bs, seq_len, 1]
        dsigma = unsqueeze_as(dsigma, score)
        dim = score.shape[-1]

        # --- Absorbing branch transformation (Qₐ) ---
        score_abs = score.clone()
        extra_const = (1 - dsigma.exp().squeeze(-1)) * score_abs.sum(dim=-1)
        score_abs = score_abs * dsigma.exp()
        score_abs[..., -1] += extra_const

        # --- Uniform branch transformation (Qᵤ) ---
        # Compute exp(-dsigma) with shape [bs, seq_len, 1]
        epow = (-dsigma).exp()

        # For nonabsorbing channels, sum only over channels 0..dim-2 and normalize by (dim-1)
        score_uni_nonabs = ((epow - 1) / ((dim - 1) * epow)) * score[..., :dim-1].sum(dim=-1, keepdim=True) \
                            + score[..., :dim-1] / epow
        # Concatenate so that score_uni has shape [bs, seq_len, dim]
        score_uni = torch.cat([score_uni_nonabs, score[..., -1:]], dim=-1)

        # --- Combine in a convex combination ---
        return (1 - self.gamma) * score_abs + self.gamma * score_uni