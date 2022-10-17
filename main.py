import math
import pickle
import torch
import argparse
import torch.nn.functional as F

from models.unet import Unet
from utils.dataloader import Dataloader
from utils.forward_diffusion import forward_diffusion
from utils.diffusion_gif import reverse_diffusion_gif, forward_diffusion_gif


IMG_SIZE = 32
BATCH_SIZE = 128

def train(epochs, lr, T, schedule, debug, save_model, gif, model_name):
    """
    Trains the model to predict noise at a timestep
    
    :params epochs: Number of epochs to train for (int)
    :params lr: Learning rate (float)
    :params T: Number of timesteps (int)
    :params schedule: Variance schedules (str)
    :params debug: For debugging (bool)
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'------------------------------Using {device}------------------------------') 
    model = Unet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    data = Dataloader(IMG_SIZE, BATCH_SIZE, debug=debug)
    print('Total number of batches in the datset', len(data))
        
    for epoch in range(epochs):
        for step, batch in enumerate(data):
            optimizer.zero_grad()
            
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            x_noisy, true_noise = forward_diffusion(batch[0].to(device), t, T, device, schedule)
            predicted_noise = model(x_noisy.to(device), t)
            
            loss = F.l1_loss(predicted_noise, true_noise)
            
            loss.backward()
            optimizer.step()
            if epoch % 1 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()} ")

    if gif:
        print(f'------------------------------Saving forward and reverse diffusion gifs------------------------------')
        forward_diffusion_gif(batch[0][0], T, IMG_SIZE, device, schedule)
        reverse_diffusion_gif(model, T, IMG_SIZE, device, schedule)
        
    if save_model:
        print(f'------------------------------Saving model as {model_name}------------------------------')
        pickle.dump(model, open(model_name, 'wb'))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--T', type=int, default=300, help='Number of timesteps')
    parser.add_argument('--schedule', type=str, default='linear',
                        help='Variance schedules', choices=['linear', 'cosine'])
    parser.add_argument('--debug', type=bool, default=False, help='For debugging')
    parser.add_argument('--save_model', type=bool, default=False, help='To save trained model')
    parser.add_argument('--gif', type=bool, default=False, help='Save forward and reverse diffusion gif')
    parser.add_argument('--model_name', type=str, default='model', help='Model name - for saving')
    
    args = parser.parse_args()
    train(args.epochs, args.lr, args.T, args.schedule, args.debug, args.save_model, args.gif, args.model_name)
