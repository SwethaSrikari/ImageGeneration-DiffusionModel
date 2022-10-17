import math
import torch
from torch import nn


def positional_encodings(t, dim, device):
    """
    To enable the model to learn the noise added at each timestep,
    a positional encoding along with the input is fed as input
    
    Reference - https://huggingface.co/blog/annotated-diffusion#position-embeddings
    
    :params t: time steps - a tensor 
    :params dim: encodings dimension - a number (int)
    :params device: cuda or cpu (str)
    
    Returns embeddings - shape(len(t), dim)
    """
    half_dim = dim // 2
    embeddings = math.log(10000) / (half_dim)
    embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
    embeddings = t[:, None] * embeddings[None, :].to(device)
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    # For alternating sine and cosine embeddings
    # Ref - https://stackoverflow.com/questions/71628542/how-to-alternatively-concatenate-pytorch-tensors
    embeddings = embeddings.T.flatten()
    embeddings = torch.stack(torch.split(embeddings, len(t)), dim=1).reshape(len(t),-1)
    
    return embeddings.to(device)


class Block(nn.ModuleDict):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up: # Upsampling (Decoder)
            self.conv2d1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1) # Same size, half the no. of channels
            self.conv2d2 = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) # Double size, same no. of channels
            self.conv2dl = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1) # Same size, same no. of channels
        else: # Downsampling (Encoder)
            self.conv2d1 = nn.Conv2d(in_ch, out_ch, 3, padding=1) # Same size, double the no. of channels
            self.conv2d2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1) # Same size, same no. of channels
            self.conv2dl = nn.Conv2d(out_ch, out_ch, 4, 2, 1) # Half size, same no. of channels
        self.relu = torch.nn.ReLU()
        self.conv2d3 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1) # Same size, same no. of channels
        self.conv2d4 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1) # Same size, same no. of channels
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, t):
        """
        Block performs a series of convolutions that increase/decrease
        the spatial dimension and the number of channels in both 
        downsampling (Encoder) and upsampling (Decoder) stages
        
        :params x: input
        :params t: time embeddings
        """
        c1 = self.bnorm1(self.relu(self.conv2d1(x)))
        
        # Time embedding - 2D
        t_emb = self.relu(self.time_mlp(t)) # shape(batch_size, out_ch)
        # Extend time embedding to 4D like x
        t_emb = t_emb.unsqueeze(2).unsqueeze(3) # shape(batch_size, out_ch, 1, 1)

        c2 = self.relu(self.conv2d2(c1)) # shape(batch_size, out_ch, size, size)
        
        # Add time embedding to up-convolved input
        c2 = c2 + t_emb # shape(batch_size, out_ch, size, size)
        
        c3 = self.bnorm2(self.relu(self.conv2d3(c2)))
        c4 = self.relu(self.conv2d4(c3))
        cl = self.conv2dl(c4)

        return cl
    
    
class Encoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 1024), time_emb_dim=32):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([Block(channels[c], channels[c+1],
                                                   time_emb_dim) for c in range(len(channels)-1)])

    def forward(self, x, t):
        """
        Encoder doubles the number of channels in the first convolution,
        followed by a series of convolutions that neither change size nor no. of channels,
        halves the input size (spatial dimension) at the final convolution
        and saves the outputs to concatenate during upsampling. 
        It is repeated until the number of channels reaches max. no. of channels
        
        :params x: input, shape(batch_size, enc[0], original i/p size, original i/p size)
        :params t: time embeddings, shape(batch_size, self.tim_dim)
        
        Returns output, shape(batch_size, enc[-1], size/len(enc), size/len(enc)) and a list of encoder o/ps
        """
        enc_ops = []
        for b, block in enumerate(self.encoder_blocks):
            x = block(x, t)
            enc_ops.append(x) # Save encoder o/p
    
        return x, enc_ops
    

class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64), time_emb_dim=32):
        super(Decoder, self).__init__()
        self.channels = channels
        self.decoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1],
                                                   time_emb_dim, up=True) for i in range(len(channels)-1)])

    def forward(self, x, enc_ops, t):
        """
        Decoder halves the number of channels in the first convolution step,
        doubles the input size in the second convolution step,
        followed by a series of convolutions that neither change size nor no. of channels.
        It is repeated until the number of channels reaches min. no. of channels
        
        :params x: final encoder output, shape(batch_size, enc[-1], size/len(enc), size/len(enc))
        :params enc_ops: list of encoder outputs
        :params t: time embeddings, shape(batch_size, self.tim_dim)
        
        Returns output, shape(batch_size, dec[-1], original i/p size, original i/p size)
        """
        for i in range(len(self.channels)-1):
            enc = enc_ops[i]
            x = torch.cat((enc, x), 1) # Concatenate encoder o/p (on channels dim, so doubles the no. of channels)
            x = self.decoder_blocks[i](x, t)

        return x
    
    
class Unet(nn.Module):
    def __init__(self, device):
        super(Unet, self).__init__()
        enc_ch=(64, 128, 256, 512, 1024) # Encoder channels
        dec_ch=(1024, 512, 256, 128, 64) # Decoder channels
        image_channels = 1 # Number of i/p channels
        self.tim_dim = 256 # time embedding dimension
        self.device = device # cuda or cpu

        # Time embedding
        self.time_mlp = nn.Sequential(
                                      nn.Linear(self.tim_dim, self.tim_dim),
                                      nn.ReLU()
                                      ).to(self.device)

        self.conv0 = nn.Conv2d(image_channels, enc_ch[0], 3, padding=1).to(self.device) # 3 channels to enc[0] channels
        self.encoder = Encoder(channels=enc_ch, time_emb_dim=self.tim_dim).to(self.device)
        self.decoder = Decoder(channels=dec_ch, time_emb_dim=self.tim_dim).to(self.device)
        self.conv1 = nn.Conv2d(dec_ch[-1], image_channels, 3, padding=1).to(self.device) # dec[-1] channels to 3 channels

    def forward(self, x, timestep):
        """
        Initial convolution - 3 channels to enc[0] channels,
        followed by encoder and decoder operations and
        final convolution - dec[-1] channels to 3 channels
        
        :params x: images at specific time steps, with(t=1toT)/without(t=0) noise added,
                   shape(batch_size, channels, img_size, img_size)
        :params timestep: a tensor of time steps, shape(batch_size)
        
        Returns noise (predicted) at specific timesteps shape(x)
        """
        # Time embeddings
        t = positional_encodings(timestep, self.tim_dim, self.device)
        t = self.time_mlp(t) # shape(batch_size, self.tim_dim)

        # Initial convolution
        x = self.conv0(x)
        # Encoder / down-sampling
        res, enc_ops = self.encoder(x, t)
        # Decoder / Up-sampling
        out = self.decoder(res, enc_ops[::-1], t)
        
        # Final convolution - Set output channels to desired number
        out = self.conv1(out)
        return out # shape(x)