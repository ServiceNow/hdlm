import abc
import torch
import torch.nn as nn


def get_noise(config):
    # check if config.noise.ar_diffusion even exists in the config:

    if config.noise.type == "geometric":
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    elif config.noise.type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"{config.noise.type} is not a valid noise")


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """
    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass


class GeometricNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=5e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)
        
    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)


class ARDiffusionNoise(Noise):
    """
    Noise scheduler implementing AR-DIFFUSION.
    """

    def __init__(self, config):
        super().__init__()
        self.max_timesteps = 1000  # T
        self.max_seq_length = 1024       # N
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anchor_point = (2 * self.max_seq_length, self.max_timesteps)  # (n_e, t_e)

        # Base noise schedule (e.g., geometric)
        self.base_noise = GeometricNoise(config.noise.sigma_min, config.noise.sigma_max) if config.noise.type == "geometric" else LogLinearNoise()

    def compute_token_timesteps(self, batch_size, seq_len):
        """
        Computes token-level timesteps f(n, t) for each token in the batch.
        """
        # Sample sentence-level timestep t from [0, N + T]
        t = torch.rand(batch_size, device=self.device) * (self.max_seq_length + self.max_timesteps)

        # Compute start point (n_s, t_s)
        n_s = torch.clamp(self.max_seq_length - t, min=0, max=self.max_seq_length)
        t_s = torch.clamp(t - self.max_seq_length, min=0, max=self.max_timesteps)

        # Anchor point (n_e, t_e)
        n_e, t_e = self.anchor_point

        # Prepare token positions n (1 to seq_len)
        n = torch.arange(1, seq_len + 1, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

        # Compute f(n, t) using the linear function
        numerator = (t_e - t_s.unsqueeze(1))  # [batch_size, 1]
        denominator = (n_e - n_s.unsqueeze(1))  # [batch_size, 1]
        slope = numerator / denominator  # [batch_size, 1]

        f_n_t = slope * (n - n_s.unsqueeze(1)) + t_s.unsqueeze(1)  # [batch_size, seq_len]
        f_n_t = torch.clamp(f_n_t, min=0, max=self.max_timesteps)

        # Normalize f_n_t to [0, 1] for compatibility with base noise scheduler
        f_n_t_normalized = f_n_t / self.max_timesteps

        return f_n_t_normalized  # [batch_size, seq_len], values in [0, 1]

    def rate_noise(self, t):
        # t: [batch_size, seq_len], values in [0, 1]
        return self.base_noise.rate_noise(t)

    def total_noise(self, t):
        # t: [batch_size, seq_len], values in [0, 1]
        return self.base_noise.total_noise(t)

    def forward(self, t):
        """
        Overrides the forward method to compute per-token total noise and rate noise.
        """
        total_noise = self.total_noise(t)  # [batch_size, seq_len]
        rate_noise = self.rate_noise(t)    # [batch_size, seq_len]
        return total_noise, rate_noise
