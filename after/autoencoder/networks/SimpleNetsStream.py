from math import floor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import pack, rearrange, reduce, unpack
from einops_exts import rearrange_many
from torch import Tensor
from functools import reduce
from torch.nn.utils import weight_norm
import numpy as np
import cached_conv as cc
import gin

from ..core import SnakeBeta as Snake
from ..core import mod_sigmoid, SimpleLatentReg
from .pqmf import PQMF, CachedPQMF


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


def Conv1d(*args, **kwargs) -> nn.Module:
    return cc.Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return cc.ConvTranspose1d(*args, **kwargs)


def Downsample1d(in_channels: int,
                 out_channels: int,
                 factor: int,
                 kernel_multiplier: int = 2,
                 cumulative_delay: int = 0) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return normalization(
        Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * kernel_multiplier,
            stride=factor,
            padding=cc.get_padding(2 * factor,
                                   factor),  #math.ceil(factor / 2),
            cumulative_delay=cumulative_delay,
        ))


def Upsample1d(in_channels: int,
               out_channels: int,
               factor: int,
               cumulative_delay: int = 0) -> nn.Module:
    if factor == 1:
        return normalization(Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=1),
                             cumulative_delay=cumulative_delay)
    return normalization(
        ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2,  #factor // 2 + factor % 2,
            # output_padding=factor % 2,
            cumulative_delay=cumulative_delay,
        ))


def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        # The hook we want to remove is an instance of WeightNorm class, so
        # normally we would do `if isinstance(...)` but this class is not accessible
        # because of shadowing, so we check the module name directly.
        # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            torch.nn.utils.remove_weight_norm(self)
    return self


def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')


@gin.configurable
class CachedGroupNorm(nn.Module):

    def __init__(self,
                 num_groups,
                 num_channels,
                 padding="automatic",
                 stream=False,
                 **kwargs):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, **kwargs)
        self.padding = padding
        self.initialized = 0
        self.stream = stream
        self.padding_size = 0
        self.register_buffer('pad', torch.zeros((4, num_channels, 1)))

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x: torch.Tensor) -> None:
        shape = x.shape
        if len(shape) > 3:
            b, c, h, t = x.shape
            if self.padding == "automatic":
                self.padding_size = t
            self.register_buffer(
                'pad',
                torch.zeros((4, c, h, self.padding_size)).to(x))
        else:
            b, c, t = x.shape
            if self.padding == "automatic":
                self.padding_size = t
            self.register_buffer('pad',
                                 torch.zeros((4, c, self.padding_size)).to(x))
        print("Initialized CachedGroupNorm with padding of shape",
              self.pad.shape)
        self.initialized += 1

    def forward(self, x):
        if self.stream:
            in_shape = x.shape[-1]
            if not self.initialized:
                self.init_cache(x)

            x = torch.cat([self.pad[:x.shape[0]], x], -1)
            # x = x[..., -self.padding_size:]
            self.pad[:x.shape[0]].copy_(x[..., -self.padding_size:])

            x = self.gn.forward(x)
            return x[..., -in_shape:]

        else:
            return self.gn.forward(x)


class ConvBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 num_groups: int = 8,
                 use_norm: bool = True,
                 activation: nn.Module = Snake,
                 cumulative_delay=0):
        super().__init__()

        groupnorm = (CachedGroupNorm(num_groups=min(in_channels, num_groups),
                                     num_channels=in_channels)
                     if use_norm else nn.Identity())

        activation = activation(dim=in_channels)
        project = normalization(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                #padding="same",
                padding=cc.get_padding(kernel_size, dilation=dilation),
                dilation=dilation,
                cumulative_delay=cumulative_delay,
            ))

        net = [groupnorm, activation, project]
        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[-1].cumulative_delay

    def __prepare_scriptable__(self):
        for hook in self.net[-1]._forward_pre_hooks.values():
            if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                #print("Removing weight_norm from %s", self.__class__.__name__)
                torch.nn.utils.remove_weight_norm(self.net[-1])
        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResnetBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 use_norm: bool = True,
                 num_groups: int = 8,
                 use_res=True,
                 activation: nn.Module = Snake,
                 cumulative_delay=0) -> None:
        super().__init__()

        self.use_res = use_res
        block1 = ConvBlock1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=dilation,
                             use_norm=use_norm,
                             num_groups=num_groups,
                             activation=activation)

        block2 = ConvBlock1d(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             use_norm=use_norm,
                             activation=activation)

        to_out = (normalization(
            Conv1d(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=1))
                  if in_channels != out_channels else nn.Identity())

        net = [block1, block2]
        additional_delay = net[0].cumulative_delay
        net = cc.CachedSequential(*net)

        if self.use_res:
            self.net = cc.AlignBranches(
                net,
                to_out,
                delays=[additional_delay, 0],
            )
            self.cumulative_delay = additional_delay + cumulative_delay

        else:
            self.net = net
            self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x: Tensor) -> Tensor:

        x, xres = self.net(x)
        return x + xres


class ResnetBlock1dNoRes(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 use_norm: bool = True,
                 num_groups: int = 8,
                 use_res=True,
                 activation: nn.Module = Snake,
                 cumulative_delay=0) -> None:
        super().__init__()

        self.use_res = use_res
        block1 = ConvBlock1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=dilation,
                             use_norm=use_norm,
                             num_groups=num_groups,
                             activation=activation)

        block2 = ConvBlock1d(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             use_norm=use_norm,
                             activation=activation)

        net = [block1, block2]
        additional_delay = net[0].cumulative_delay
        net = cc.CachedSequential(*net)

        self.net = net
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return y


class DownsampleBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 factor: int,
                 num_groups: int,
                 num_layers: int,
                 dilations: Sequence[int],
                 kernel_size: int,
                 activation: nn.Module,
                 use_norm: bool,
                 cumulative_delay=0):
        super().__init__()

        net = []
        for i in range(num_layers):
            net.append(
                ResnetBlock1d(in_channels=in_channels,
                              out_channels=in_channels,
                              num_groups=num_groups,
                              activation=activation,
                              dilation=dilations[i],
                              kernel_size=kernel_size,
                              use_norm=use_norm,
                              cumulative_delay=(net[-1].cumulative_delay if i
                                                > 0 else cumulative_delay)))

        net.append(activation(in_channels))
        net.append(
            Downsample1d(in_channels=in_channels,
                         out_channels=out_channels,
                         factor=factor,
                         cumulative_delay=net[-2].cumulative_delay))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UpsampleBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 factor: int,
                 num_layers: int,
                 dilations: Sequence[int],
                 kernel_size: int,
                 num_groups: int,
                 activation: nn.Module,
                 use_norm: bool,
                 cumulative_delay: int = 0):
        super().__init__()

        net = [activation(dim=in_channels)]
        upsample = Upsample1d(in_channels=in_channels,
                              out_channels=out_channels,
                              factor=factor)

        cd = upsample.cumulative_delay + cumulative_delay * factor

        net.append(upsample)
        for i in range(num_layers):
            net.append(
                ResnetBlock1d(in_channels=out_channels,
                              out_channels=out_channels,
                              num_groups=num_groups,
                              activation=activation,
                              dilation=dilations[i],
                              kernel_size=kernel_size,
                              use_norm=use_norm,
                              cumulative_delay=net[-1].cumulative_delay
                              if i > 0 else cd))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


"""
Encoders / Decoders
"""


class Bottleneck(nn.Module):

    def forward(self,
                x: Tensor,
                with_info: bool = False) -> Union[Tensor, Tuple[Tensor, Any]]:
        raise NotImplementedError()


class Encoder1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 multipliers: Sequence[int],
                 factors: Sequence[int],
                 num_blocks: Sequence[int],
                 dilations: Sequence[int],
                 kernel_size: int,
                 resnet_groups: int = 8,
                 out_channels: Optional[int] = None,
                 activation: nn.Module = Snake,
                 use_norm: bool = True):
        super().__init__()
        self.num_layers = len(multipliers) - 1
        self.downsample_factor = prod(factors)
        self.out_channels = out_channels
        assert len(factors) == self.num_layers and len(
            num_blocks) == self.num_layers

        to_in = ResnetBlock1d(in_channels=in_channels,
                              out_channels=channels * multipliers[0],
                              use_res=True,
                              activation=activation,
                              use_norm=use_norm,
                              kernel_size=kernel_size)

        net = [to_in]

        for i in range(self.num_layers):
            net.append(
                DownsampleBlock1d(in_channels=channels * multipliers[i],
                                  out_channels=channels * multipliers[i + 1],
                                  factor=factors[i],
                                  num_groups=resnet_groups,
                                  num_layers=num_blocks[i],
                                  dilations=dilations,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  use_norm=use_norm,
                                  cumulative_delay=net[-1].cumulative_delay))

        #net.append(recurent_layer(
        #    in_size=channels * multipliers[-1],
        #    out_size=channels * multipliers[-1]))

        net.append(activation(dim=channels * multipliers[-1]))
        net.append(
            normalization(
                cc.Conv1d(in_channels=channels * multipliers[-1],
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=cc.get_padding(kernel_size=3, dilation=1),
                          cumulative_delay=net[-2].cumulative_delay)))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def amp_to_impulse_response(amp, target_size):
    """
    transforms frequency amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


class NoiseGenerator(nn.Module):

    def __init__(self,
                 in_size,
                 data_size,
                 ratios,
                 noise_bands,
                 hidden_size=128):
        super().__init__()
        net = []

        channels = [in_size]
        channels.extend((len(ratios) - 1) * [hidden_size])
        channels.append(data_size * noise_bands)

        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r),
                    stride=r,
                    cumulative_delay=cum_delay,
                ))
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(
            np.prod(ratios))

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class Decoder1d(nn.Module):

    def __init__(self,
                 out_channels: int,
                 channels: int,
                 multipliers: Sequence[int],
                 factors: Sequence[int],
                 num_blocks: Sequence[int],
                 dilations: Sequence[int],
                 kernel_size: int,
                 resnet_groups: int = 8,
                 in_channels: Optional[int] = None,
                 recurent_layer: nn.Module = nn.Identity,
                 activation: nn.Module = Snake,
                 use_norm: bool = True,
                 use_loudness: bool = False,
                 use_noise: bool = False):
        super().__init__()

        num_layers = len(multipliers) - 1

        assert len(factors) == num_layers and len(num_blocks) == num_layers

        self.use_loudness = use_loudness
        self.use_noise = use_noise

        net = []

        #if recurrent_layer is not None:
        #    recurrent_layer = recurent_layer(in_size=in_channels,
        #                                        out_size=in_channels)
        #
        #    net.append(recurrent_layer)

        to_in = normalization(
            Conv1d(in_channels=in_channels,
                   out_channels=channels * multipliers[0],
                   kernel_size=kernel_size,
                   padding=cc.get_padding(kernel_size, dilation=1)
                   #padding="same"
                   ))

        net.append(to_in)

        for i in range(num_layers):
            net.append(
                UpsampleBlock1d(in_channels=channels * multipliers[i],
                                out_channels=channels * multipliers[i + 1],
                                factor=factors[i],
                                num_groups=resnet_groups,
                                num_layers=num_blocks[i],
                                dilations=dilations,
                                activation=activation,
                                kernel_size=kernel_size,
                                use_norm=use_norm,
                                cumulative_delay=net[-1].cumulative_delay))

        to_out = ResnetBlock1dNoRes(in_channels=channels * multipliers[-1],
                                    out_channels=out_channels *
                                    2 if self.use_loudness else out_channels,
                                    use_res=False,
                                    activation=activation,
                                    use_norm=use_norm,
                                    kernel_size=kernel_size,
                                    cumulative_delay=net[-1].cumulative_delay)

        self.net = cc.CachedSequential(*net)

        branches = [to_out]

        if self.use_noise:
            self.noise_module = NoiseGenerator(in_size=channels *
                                               multipliers[-1],
                                               data_size=out_channels,
                                               ratios=[2, 2, 2],
                                               noise_bands=5)
            branches.append(self.noise_module)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)

        if self.use_noise:
            x, noise = self.synth(x)
        else:
            noise = torch.tensor(0.)
            x = self.synth(x)[0]

        if self.use_loudness:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        if self.use_noise:
            x = x + noise

        return x


"""
def gaussian_sample(mean: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    sample = mean + std * eps
    return sample


def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
    losses = mean**2 + logvar.exp() - logvar - 1
    loss = reduce(losses, "b ... -> 1", "mean").item()
    return loss


class VariationalBottleneck(Bottleneck):
    def __init__(self, channels: int, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.to_mean_and_std = Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
        )

    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        mean_and_std = self.to_mean_and_std(x)
        mean, std = mean_and_std.chunk(chunks=2, dim=1)
        mean = torch.tanh(mean)  # mean in range [-1, 1]
        std = torch.tanh(std) + 1.0  # std in range [0, 2]
        out = gaussian_sample(mean, std)
        info = dict(
            variational_kl_loss=kl_loss(mean, std) * self.loss_weight,
            variational_mean=mean,
            variational_std=std,
        )
        return (out, info) if with_info else out


class TanhBottleneck(Bottleneck):
    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        x = torch.tanh(x)
        info: Dict = dict()
        return (x, info) if with_info else x


class NoiserBottleneck(Bottleneck):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        if self.training:
            x = torch.randn_like(x) * self.sigma + x
        info: Dict = dict()
        return (x, info) if with_info else x
"""


@gin.configurable
class TanhBottleneck(nn.Module):

    def __init__(self,
                 scale: int = 3,
                 sigma: float = 0.,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.scale = scale

    def forward(self,
                x: Tensor,
                apply_noise: bool = True) -> Tuple[Tensor, Tensor]:

        if apply_noise:
            x = self.scale * torch.tanh(x) + self.sigma * torch.randn_like(x)
        else:
            x = self.scale * torch.tanh(x)

        return x, torch.tensor(0.)


class ReluBottleneck(nn.Module):

    def __init__(self,
                 scale: int = 3,
                 sigma: float = 0.,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.reg_loss = SimpleLatentReg(scale=scale)

    def forward(self,
                x: Tensor,
                apply_noise: bool = False) -> Tuple[Tensor, Tensor]:

        reg_loss = self.reg_loss(x)
        if apply_noise:
            x = x + self.sigma * torch.randn_like(x)
        return x, reg_loss


class VAEBottleneck(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        z: Tensor,
        return_mean: bool = False,
    ) -> Tensor:

        mean, scale = z.chunk(2, 1)

        std = nn.functional.softplus(scale) + 1e-2
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        if return_mean:
            return z, kl, mean
        return z, kl


class GRU(nn.Module):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.enabled = True

        self.to_out = normalization(
            cc.Conv1d(hidden_size,
                      out_size,
                      kernel_size=3,
                      padding=cc.get_padding(3, dilation=1)))  #padding same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.gru(x)[0]
        x = x.permute(0, 2, 1)
        x = self.to_out(x)
        return x


class DummyIdentity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


@gin.configurable
class AutoEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 z_channels: int,
                 multipliers: Sequence[int],
                 factors: Sequence[int],
                 dilations: Sequence[int],
                 kernel_size: int,
                 resnet_groups: int = 8,
                 bottleneck: nn.Module = nn.Identity,
                 activation: nn.Module = Snake,
                 use_norm: bool = True,
                 decoder_ratio: int = 1,
                 pqmf_bands: int = 0,
                 use_loudness: bool = False,
                 use_noise: bool = False):
        super().__init__()
        out_channels = in_channels

        self.pqmf_bands = pqmf_bands
        if self.pqmf_bands > 1:
            self.pqmf = CachedPQMF(attenuation=100, n_band=pqmf_bands)
            self.use_pqmf = True
        else:
            self.pqmf = DummyIdentity()
            self.use_pqmf = False

        num_blocks = [3] * len(factors)

        #adapt channels if vaebottleneck is used
        if isinstance(bottleneck, VAEBottleneck):
            encoder_out_channels = z_channels * 2
        else:
            encoder_out_channels = z_channels

        self.encoder = Encoder1d(in_channels=in_channels,
                                 out_channels=encoder_out_channels,
                                 channels=channels,
                                 multipliers=multipliers,
                                 factors=factors,
                                 num_blocks=num_blocks,
                                 dilations=dilations,
                                 kernel_size=kernel_size,
                                 resnet_groups=resnet_groups,
                                 activation=activation,
                                 use_norm=use_norm)

        self.decoder = Decoder1d(
            in_channels=z_channels,
            out_channels=out_channels,
            channels=channels,
            multipliers=[int(m * decoder_ratio) for m in multipliers[::-1]],
            factors=factors[::-1],
            num_blocks=num_blocks[::-1],
            dilations=dilations,
            kernel_size=kernel_size,
            resnet_groups=resnet_groups,
            activation=activation,
            use_norm=use_norm,
            use_loudness=use_loudness,
            use_noise=use_noise)

        self.bottleneck = bottleneck

    @torch.jit.ignore
    def forward(self, x: Tensor, return_all: bool = True) -> Tensor:

        if self.pqmf_bands > 1:
            x_multiband = self.pqmf(x)
        else:
            x_multiband = x

        z = self.encoder(x_multiband)

        z, regloss = self.bottleneck(z)

        y_multiband = self.decoder(z)

        if self.pqmf_bands > 1:
            y = self.pqmf.inverse(y_multiband)

        if return_all:
            return y, y_multiband, z, regloss, x_multiband

    def encode(
        self,
        x: Tensor,
        with_multi: bool = False,
        return_mean: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if self.pqmf_bands > 1:
            x_multiband = self.pqmf(x)
        else:
            x_multiband = x

        z = self.encoder(x_multiband)

        if return_mean:
            z, regloss, mean = self.bottleneck(z, return_mean=True)
            return z, regloss, mean
        else:
            z, regloss = self.bottleneck(z)

            if with_multi:
                return z, x_multiband, regloss

            return z, regloss

    def decode(self, z: Tensor, with_multi: bool = False) -> Tensor:
        x_multiband = self.decoder(z)

        if self.pqmf_bands > 1:
            x = self.pqmf.inverse(x_multiband)
        else:
            x = x_multiband

        if with_multi:
            return x, x_multiband

        return x
