# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20

    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
        
    def get_lambda(self, t: torch.Tensor):
        # Expect t as integers in [0, n_steps), normalize to u ∈ [0,1]
        device = t.device
        dtype = t.dtype

        # Convert to float and normalize
        u = t.float() / self.n_steps
        u = u.clamp(1e-5, 1 - 1e-5)  # Stability

        lambda_min = torch.tensor(self.lambda_min, device=device, dtype=dtype)
        lambda_max = torch.tensor(self.lambda_max, device=device, dtype=dtype)

        b = torch.atan(torch.exp(-0.5 * lambda_max))
        a = torch.atan(torch.exp(-0.5 * lambda_min)) - b

        lambda_t = -2.0 * torch.log(torch.tan(a * u + b))
        return lambda_t.view(-1, 1, 1, 1)

    def alpha_lambda(self, lambda_t: torch.Tensor): 
        return 1.0 / torch.sqrt(1.0 + torch.exp(-lambda_t))
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        return 1.0 / torch.sqrt(1.0 + torch.exp(lambda_t))
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)
        z_lambda_t = alpha * x + sigma * noise
        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        sigma2_lambda_prim = 1.0 / (1.0 + torch.exp(lambda_t_prim))
        diff = lambda_t - lambda_t_prim
        weight = 1.0 - torch.exp(diff)
        var_q = weight * sigma2_lambda_prim
        return torch.sqrt(var_q)
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        sigma2_lambda_prim = 1.0 / (1.0 + torch.exp(lambda_t_prim))
        diff = lambda_t - lambda_t_prim
        weight = 1.0 - torch.exp(diff)
        var_q_x = weight * sigma2_lambda_prim
        return torch.sqrt(var_q_x)

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        alpha_t = torch.sqrt(1.0 / (1.0 + torch.exp(-lambda_t)))
        alpha_t_prim = torch.sqrt(1.0 / (1.0 + torch.exp(-lambda_t_prim)))

        exp_diff = torch.exp((lambda_t - lambda_t_prim).clamp(max=80))

        term1 = (exp_diff * (alpha_t_prim / alpha_t)) * z_lambda_t

        one_minus_exp = (1.0 - exp_diff).clamp(min=0.0)
        term2 = one_minus_exp * alpha_t_prim * x

        mu = term1 + term2
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float = 0.3):
        sigma2_lambda_prim = 1.0 / (1.0 + torch.exp(lambda_t_prim))
        weight = 1.0 - torch.exp(lambda_t - lambda_t_prim)

        var = weight * sigma2_lambda_prim
        return var.clamp(min=1e-10)

    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)

        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        noise = torch.randn_like(mu)
        sample = mu + torch.sqrt(var) * noise
        return sample

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)

        # TODO: q_sample z
        lambda_t = self.get_lambda(t)
        z_t = self.q_sample(x0, lambda_t, noise)

        # TODO: compute loss
        predicted_noise = self.eps_model(z_t, labels)
        loss = (noise - predicted_noise).pow(2).sum(dim=dim).mean()

        return loss
