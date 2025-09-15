import torch


def sample_categorical(categorical_probs, method="hard", temperature=1.0):
    if method == "hard":
        # Generate uniform random noise u
        u = torch.rand_like(categorical_probs, dtype=torch.float64) + 1e-10
        
        # Compute Gumbel noise and apply temperature as exponent
        gumbel_noise = (-torch.log(u)) ** temperature
        
        # Sample via argmax according to paper implementation
        return (categorical_probs / gumbel_noise).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    