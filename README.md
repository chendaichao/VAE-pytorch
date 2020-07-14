# VAEs for Handwritten Number Generation

Pytorch implementation for Variational AutoEncoders (VAEs) and conditional Variational AutoEncoders. 



## A short description

![A short description](vae_description.png)

## Implementation

We use *pytorch* to implement the model. We use MNIST (a dataset of handwritten digits) to train the models. The encoders and the decoder are all simple MLPs (please refer to [model.py](Models/VAE/model.py) for more details). You can also run the jupyter notebook [Demo.ipynb](Models/VAE/Demo.ipynb) to see a few illustrations.
