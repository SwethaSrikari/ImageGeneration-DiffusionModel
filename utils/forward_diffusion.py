import torch

from utils.variance_schedule import variance_schedule

def forward_diffusion(x0, t, T, device, schedule):
    """
    Math based on https://arxiv.org/abs/2006.11239
    
    :params x0: Input image - shape(batch_size, num_channels, img_size, img_size)
    :params t: time step - shape(batch_size)
    :params T: number of time steps (int)
    :params device: cuda or cpu (str)
    :params schedule: Variance schedule - 'linear' or 'cosine'
    
    Returns image corrupted with noise at a given timestep 
    along with the noise added at that time step
    """
    dim = len(x0.shape)-1
    noise = torch.randn_like(x0)

    beta = variance_schedule(T, device, schedule)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, 0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    
    x = (sqrt_alpha_bar[t].reshape(len(t), *(1,)*dim).to(device) * x0.to(device)) +\
        (sqrt_one_minus_alpha_bar[t].reshape(len(t), *(1,)*dim).to(device) * noise.to(device))
    
    return x.to(device), noise.to(device)