"""
2D time-frequency autoencoder.
Ported from acids_codecs/networks/AE2D.py.
Uses StreamableSTFT as time_transform (see after.autoencoder.audio).
"""
import math

import torch
import torch.nn as nn
import gin
import cached_conv as cc
import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from .cachedConv2D import (CachedConv2d, CachedConvTranspose2d,
                           CachedPadding2d, AlignBranches2d)


def _wn(module: nn.Module):
    layer = torch.nn.utils.weight_norm(module)

    def _prepare_scriptable(self):
        for hook in self._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self)
        return self

    layer.__prepare_scriptable__ = _prepare_scriptable.__get__(layer)
    return layer


def Conv1d(*args, **kwargs) -> nn.Module:
    return _wn(cc.Conv1d(*args, **kwargs))


def Conv2d(*args, **kwargs) -> nn.Module:
    return CachedConv2d(*args, **kwargs, normalization=True)


def ConvTranspose2d(*args, **kwargs) -> nn.Module:
    return _wn(CachedConvTranspose2d(*args, **kwargs))


class EncoderBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 kernel_size=3,
                 ratio=2,
                 freq_ratio=2,
                 kernel_multiplier=2,
                 act=nn.SiLU):
        super().__init__()
        self.act = act()
        self.conv = Conv2d(in_c,
                           in_c,
                           kernel_size=kernel_size,
                           padding_vert="same",
                           padding_time=cc.get_padding(kernel_size=kernel_size,
                                                       stride=1,
                                                       mode="causal"))
        self.pool = Conv2d(in_channels=in_c,
                           out_channels=out_c,
                           kernel_size=(freq_ratio * kernel_multiplier + 1,
                                        ratio * kernel_multiplier + 1),
                           stride=(freq_ratio, ratio),
                           padding_vert="same",
                           padding_time=cc.get_padding(
                               kernel_size=ratio * kernel_multiplier + 1,
                               mode="causal",
                               stride=ratio))
        self.proj_pool = nn.AvgPool2d(kernel_size=(freq_ratio, ratio),
                                      stride=(freq_ratio, ratio))
        self.proj = Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        res = x.clone()
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        res = self.proj_pool(res)
        res = self.proj(res)
        return x + res


class DecoderBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 kernel_size,
                 act=nn.SiLU,
                 ratio=2,
                 freq_ratio=2):
        super().__init__()
        self.act = act()
        self.up = ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(2 * freq_ratio + (1 if freq_ratio == 1 else 0),
                         2 * ratio + (1 if ratio == 1 else 0)),
            stride=(freq_ratio, ratio),
            padding=(freq_ratio // 2 + (1 if freq_ratio == 1 else 0),
                     ratio // 2 + (1 if ratio == 1 else 0)))
        self.conv = Conv2d(in_c,
                           in_c,
                           kernel_size=kernel_size,
                           padding_vert="same",
                           padding_time=cc.get_padding(kernel_size=kernel_size,
                                                       stride=1,
                                                       mode="causal"))
        self.proj_pool = nn.Upsample(scale_factor=(freq_ratio, ratio),
                                     mode='nearest')
        self.proj = Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.cumulative_delay = self.conv.cumulative_delay

    def forward(self, x):
        res = x.clone()
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.up(x)
        res = self.proj_pool(res)
        res = self.proj(res)
        return x + res


class ConvBlock1d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 activation: nn.Module = nn.SiLU,
                 cumulative_delay=0):
        super().__init__()
        self.activation = activation()
        self.project = Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=cc.get_padding(kernel_size,
                                                     dilation=dilation,
                                                     mode="causal"),
                              dilation=dilation,
                              cumulative_delay=cumulative_delay)

    def __prepare_scriptable__(self):
        for hook in self.project._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self.project)
        return self

    def forward(self, x):
        x = self.activation(x)
        x = self.project(x)
        return x


@gin.configurable
class AutoEncoder2D(nn.Module):
    """
    2D time-frequency autoencoder operating on complex spectrograms.

    time_transform should be a StreamableSTFT instance (from after.autoencoder.audio).
    Input:  (B, audio_channels, T)
    Output: (B, audio_channels, T)

    Internally processes as (B, 2, F, T) — real/imag × freq × time.
    For stereo, each channel is processed independently then merged at the
    bottleneck via stereo_merge / stereo_split blocks.
    """

    def __init__(self,
                 in_size: int = 2,
                 out_size=None,
                 bottleneck_size: int = 0,
                 audio_channels: int = 1,
                 channels=None,
                 time_ratios=None,
                 freq_ratios=None,
                 freq_size: int = 1024,
                 kernel_size: int = 3,
                 bottleneck=None,
                 time_transform=None,
                 use_vae: bool = False):
        super().__init__()

        if channels is None:
            channels = [128, 128, 128, 128]
        if time_ratios is None:
            time_ratios = [1, 1, 1, 2, 1]
        if freq_ratios is None:
            freq_ratios = [2, 2, 2, 2, 1]
        if bottleneck is None:
            bottleneck = nn.Identity()

        self.time_transform = time_transform
        self.channels = channels
        self.bottleneck = bottleneck
        self.in_size = in_size
        self.audio_channels = audio_channels
        out_size = in_size if out_size is None else out_size

        # Stereo packing/unpacking
        self.pack_audio = Rearrange('b ch t -> (b ch) 1 t', ch=audio_channels)
        self.unpack_audio = Rearrange('(b ch) 1 t -> b ch t',
                                      ch=audio_channels)
        self.merge_stereo_features = Rearrange('(b ch) c f t -> b (ch c) f t',
                                               ch=audio_channels)
        self.split_stereo_features = Rearrange('b (ch c) f t -> (b ch) c f t',
                                               ch=audio_channels)

        n = len(channels)
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.preconv = Conv2d(in_size,
                              channels[0],
                              kernel_size=7,
                              padding_vert=(7 - 1) // 2,
                              padding_time=cc.get_padding(kernel_size=7,
                                                          stride=1,
                                                          mode="causal"))
        self.outconv = Conv2d(2 * channels[0],
                              out_size,
                              kernel_size=7,
                              padding_vert=(7 - 1) // 2,
                              padding_time=cc.get_padding(kernel_size=7,
                                                          stride=1,
                                                          mode="causal"))

        for i in range(n - 1):
            self.down_layers.append(
                EncoderBlock2D(in_c=channels[i],
                               out_c=channels[i + 1],
                               kernel_size=kernel_size,
                               ratio=time_ratios[i],
                               freq_ratio=freq_ratios[i]))

        self.stereo_merge = (EncoderBlock2D(in_c=2 * channels[-1],
                                            out_c=channels[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1)
                             if audio_channels == 2 else nn.Identity())

        freq_total_ratio = math.prod(freq_ratios)
        self.freq_final_dim = freq_size // freq_total_ratio

        self.middle_block_encode = ConvBlock1d(
            in_channels=self.freq_final_dim * channels[-1],
            out_channels=bottleneck_size * 2 if use_vae else bottleneck_size,
            kernel_size=3)
        self.rearrange_encode = Rearrange('b c f t -> b (c f) t')
        self.rearrange_decode = Rearrange('b (c f) t -> b c f t',
                                          f=self.freq_final_dim)
        self.middle_block_decode = ConvBlock1d(
            in_channels=bottleneck_size,
            out_channels=2 * self.freq_final_dim * channels[-1],
            kernel_size=3)

        channels_dec = [2 * c for c in channels]

        for i in range(1, n):
            self.up_layers.append(
                DecoderBlock2D(in_c=channels_dec[n - i],
                               out_c=channels_dec[n - i - 1],
                               kernel_size=kernel_size,
                               ratio=time_ratios[n - i],
                               freq_ratio=freq_ratios[n - i]))

        self.stereo_split = (DecoderBlock2D(in_c=channels_dec[-1],
                                            out_c=2 * channels_dec[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1)
                             if audio_channels == 2 else nn.Identity())

        # Expose encoder/decoder as module lists (used by Trainer.init_opt)
        self.encoder = nn.ModuleList(
            [self.down_layers, self.middle_block_encode, self.preconv,
             self.stereo_merge])
        self.decoder = nn.ModuleList(
            [self.up_layers, self.middle_block_decode, self.outconv,
             self.stereo_split])

    def _encode_features(self, h):
        h = self.preconv(h)
        for layer in self.down_layers:
            h = layer(h)
        if self.audio_channels == 2:
            h = self.merge_stereo_features(h)
            h = self.stereo_merge(h)
        h = self.rearrange_encode(h)
        h = self.middle_block_encode(h)
        return h

    def _decode_features(self, h):
        h = self.middle_block_decode(h)
        h = self.rearrange_decode(h)
        if self.audio_channels == 2:
            h = self.stereo_split(h)
            h = self.split_stereo_features(h)
        for layer in self.up_layers:
            h = layer(h)
        h = self.outconv(h)
        return h

    @torch.jit.ignore
    def forward(self,
                x,
                return_all: bool = True,
                freeze_encoder: bool = False,
                look_ahead_steps: int = 0):
        if self.audio_channels == 2:
            x = self.pack_audio(x)

        h = self.time_transform(x)
        x_multiband = h.clone()

        if freeze_encoder:
            with torch.no_grad():
                h = self._encode_features(h)
                h, regloss = self.bottleneck(h)
        else:
            h = self._encode_features(h)
            h, regloss = self.bottleneck(h)

        z = h.clone()
        if look_ahead_steps > 0:
            z = z[..., look_ahead_steps:]
            z = torch.cat((z, torch.zeros_like(z[..., :look_ahead_steps])),
                          dim=-1)

        h = self._decode_features(z)
        y_multiband = h.clone()
        y = self.time_transform.inverse(h)

        if self.audio_channels == 2:
            y = self.unpack_audio(y)

        if return_all:
            return y, y_multiband, z, regloss, x_multiband
        return y

    @torch.jit.ignore
    def encode(self, x, with_multi: bool = False, return_mean: bool = False):
        if self.audio_channels == 2:
            x = self.pack_audio(x)
        x_multiband = self.time_transform(x)
        h = self._encode_features(x_multiband.clone())
        h, regloss = self.bottleneck(h)
        if with_multi:
            return h, x_multiband
        return h, regloss

    @torch.jit.ignore
    def decode(self, z, with_multi: bool = False):
        h = self._decode_features(z)
        y = self.time_transform.inverse(h)
        if self.audio_channels == 2:
            y = self.unpack_audio(y)
        if with_multi:
            return y, h
        return y

    @torch.jit.export
    def encode_stream(self, x):
        if self.audio_channels == 2:
            x = self.pack_audio(x)
        x = self.time_transform(x)
        h = self._encode_features(x)
        h = self.bottleneck.forward_stream(h)
        return h

    @torch.jit.export
    def decode_stream(self, z):
        h = self._decode_features(z)
        y = self.time_transform.inverse_stream(h)
        if self.audio_channels == 2:
            y = self.unpack_audio(y)
        return y

    @torch.jit.export
    def forward_stream(self, x):
        if self.audio_channels == 2:
            x = self.pack_audio(x)
        x = self.time_transform(x)
        h = self._encode_features(x)
        h = self.bottleneck.forward_stream(h)
        h = self._decode_features(h)
        y = self.time_transform.inverse_stream(h)
        if self.audio_channels == 2:
            y = self.unpack_audio(y)
        return y