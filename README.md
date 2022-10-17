# Image generation using a diffusion model

Diffusion models are generative models. They gradually corrupt the input by adding noise until the input completely becomes noise (forward diffusion) and try to recover the input from noise by training a model to learn the noise added to the input at each time step (reverse diffusion).

# Visualizing forward diffusion process

Visualizing noise added to an image of digit 4

![Forward diffusion](/images/forward.gif)

# Visualizing reverse diffusion process

Generating digit 7 from noise

![Reverse diffusion](/images/reverse_diffusion.gif)


# To run this script

This script 

1. trains a simple Unet model for specified number of epochs
2. the number of timesteps T, variance schedule and learning rate can be adjusted
3. the trained model can be saved
4. forward and reverse diffusion processes can be visualized using a gif

by using appropriate arguments

```
$ python main.py --epochs <Number of epochs to train for> --lr <Learning rate> --T <Number of timesteps> --schedule <Variance schedules> --debug <For debugging> --save_model <To save trained model> --gif <Save forward and reverse diffusion gif> --model_name <Model name - for saving>
```

### [Visualizing diffusion process and training a model on notebook](https://github.com/SwethaSrikari/ImageGeneration-DiffusionModel/blob/main/Foward%20and%20reverse%20diffusion%20process.ipynb)

# References

Denoising diffusion probablitistic models (DDPM) - https://arxiv.org/abs/2006.11239

Improved Denoising diffusion probablitistic models - https://openreview.net/forum?id=-NEXDKk8gZ

Speed-up sampling using Denoising diffusion implicit model (DDIM) - https://arxiv.org/abs/2010.02502

Unet - https://amaarora.github.io/2020/09/13/unet.html

Positional encoding - https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

Diffusion models -

https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

https://huggingface.co/blog/annotated-diffusion

https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=BIc33L9-uK4q
