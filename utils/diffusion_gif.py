import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.sampling import sample_loop
from utils.forward_diffusion import forward_diffusion

def reverse_diffusion_gif(model, T, IMG_SIZE, device, schedule):
    """
    Saves a gif showing the reverse diffusion process from t=T to t=0
    
    :params model: A trained model
    :params T: timesteps (integer)
    :params batch_size: Batch size
    :params IMG_SIZE: image size (height/width)
    :params device: cpu or cuda
    :params schedule: Variance schedule (str)
    """
    
    samples = sample_loop(model, T, IMG_SIZE, device, schedule)
    fig = plt.figure()
    ims = []
    for i in range(T):
        im = plt.imshow(samples[i][0].reshape(IMG_SIZE, IMG_SIZE, 1), cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('reverse_diffusion.gif')
    
def forward_diffusion_gif(batch, T, IMG_SIZE, device, schedule):
    """
    Saves a gif showing the forward diffusion process from t=0 to t=T
    :params batch: a batch of images
    :params T: timesteps (integer)
    :params IMG_SIZE: image size (height/width)
    :params device: cpu or cuda
    :params schedule: Variance schedule (str)
    """

    fig = plt.figure()
    ims = []
    for i in range(T):
        x, _ = forward_diffusion(batch, torch.tensor(i).view(1,-1), T, device, schedule)
        im = plt.imshow(x[0].reshape(IMG_SIZE, IMG_SIZE, 1).cpu().numpy(), cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('forward_diffusion.gif')