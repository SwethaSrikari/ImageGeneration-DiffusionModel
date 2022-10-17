import torch

from utils.variance_schedule import variance_schedule

@torch.no_grad()
def sampling(xt, t, T, model, device, schedule):
    """
    Uses trained model to gradually denoise the noisy image (at t) and
    recover image (at t-1)
    
    :params xt: noisy image at timestep t # shape(batch_size, channels, img_size, img_size)
    :params t: timesteps, a tensor # shape(batchsize)
    :params T: number of timesteps (int)
    :params model: Trained model
    :params schedule: Variance schedule
    
    Returns image at timestep t-1
    """
    noise = torch.randn_like(xt)
    
    beta = variance_schedule(T, device, schedule=schedule)
    alpha = 1 - beta
    one_minus_alpha = 1 - alpha
    alpha_bar = torch.cumprod(alpha, 0)
    alpha_bar_prev = torch.cat([torch.tensor([1]).cuda(), alpha_bar[:-1]], 0) #concat 1 in the beginning to match size
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    variance = ((1 - alpha_bar_prev) / (1 - alpha_bar)) * beta

    mean = (1 / torch.sqrt(alpha[t])) * (xt - (beta[t]/sqrt_one_minus_alpha_bar[t])*model(xt, t)) 
    
    if t[0] == 0:
        return mean
    else:
        x_prev = mean + torch.sqrt(variance[t]) * noise
        return x_prev
    
    
@torch.no_grad()
def sample_loop(model, T, IMG_SIZE, device, schedule):
    """
    Returns a list of denoised images from t=T to t=0
    
    :params model: A trained model
    :params T: timesteps (integer)
    :params IMG_SIZE: image size (height/width)
    :params device: cpu or cuda
    """
    xt = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
    imgs = []
    for t in range(T-1, -1, -1):
        t = torch.full((1,), t, device=device, dtype=torch.long)
        xt = sampling(xt, t, T, model, device, schedule)
        imgs.append(xt.cpu().numpy())
    return imgs