# Image generation using a diffusion model

Diffusion models are generative models. They gradually corrupt the input by adding noise until the input completely becomes noise (forward diffusion) and try to recover the input from noise by training a model to learn the noise added to the input at each time step (reverse diffusion).

# Visualizing forward diffusion process

Visualizing noise added to an image of digit 4

![Forward diffusion](/images/forward.gif)

# Visualizing reverse diffusion process

Generating digit 7 from noise

![Reverse diffusion](/images/reverse_diffusion.gif)




