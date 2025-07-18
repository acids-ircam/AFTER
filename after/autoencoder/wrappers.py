from music2latent import EncoderDecoder
import numpy as np
import torch.nn.functional as F


class M2LWrapper():
    """Wrapper for the EncoderDecoder model to use it with the AudioExample class."""

    def __init__(self, device="cpu"):
        self.model = EncoderDecoder(device=device)

    def to(self, device):
        self.model = EncoderDecoder(device=device)
        return self

    def cpu(self):
        self.model = EncoderDecoder(device="cpu")
        return self

    def eval(self):
        return self

    def encode(self, x):
        x = x.squeeze(1)
        x_padded = F.pad(x, (0, 1536))  # pad left and right

        return self.model.encode(x_padded)

    def decode(self, z):
        x = self.model.decode(z).unsqueeze(1)
        x = x[..., :-1536]  # remove padding
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))
