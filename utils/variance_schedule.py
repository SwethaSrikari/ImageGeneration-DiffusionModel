import torch

def cosine_variance_schedule(T, device):
    """
    Cosine variance schedule from paper - https://openreview.net/forum?id=-NEXDKk8gZ

    :params T: number of time steps (int)
    :params device: cuda or cpu (str)
    
    Returns beta (variance schedule) of length T
    """
    
    s=0.008
    def f(T):
        t = torch.linspace(0, T, T+1).to(device) # 0-T
        f = torch.cos((t/T + s)/(1+s) * torch.pi/2)**2
        return f 
    
    f_t = f(T)
    alpha_bar = f_t
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1] # beta_1 to beta_T
    
    return torch.clip(beta, 0.0001, 0.999).to(device)

def linear_variance_schedule(T, device):
    """
    Linear variance schedule from https://arxiv.org/abs/2006.11239
    
    :params T: number of time steps (int)
    :params device: cuda or cpu (str)
    
    Returns beta (variance schedule) of length T
    
    """
    return torch.linspace(0.0001, 0.02, T).to(device)

def variance_schedule(T, device, schedule='linear'):
    """
    :params T: number of time steps (int)
    :params schedule: Variance schedule (str)
    :params device: cuda or cpu (str)
    """
    if schedule == 'linear':
        return linear_variance_schedule(T, device)
    elif schedule == 'cosine':
        return cosine_variance_schedule(T, device)
    else:
        raise ValueError('Wrong schedule')