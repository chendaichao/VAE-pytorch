# VAEs for Handwritten Number Generation

Pytorch implementation for Variational AutoEncoders (VAEs) and conditional Variational AutoEncoders. 



## A short description

![A short description](vae_description.png)

## Implementation

We use *pytorch* to implement the model. We use MNIST (a dataset of handwritten digits) to train the models. The encoders $\mu_\phi, \log \sigma^2_\phi$ are shared convolutional networks followed by their respective MLPs. The decoder is a simple MLP. Please refer to [model.py](Models/VAE/model.py) for more details. You can also run the jupyter notebook [Demo.ipynb](Models/VAE/Demo.ipynb) to see a few illustrations.
