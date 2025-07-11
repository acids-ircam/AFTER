import torch
from einops.layers.torch import Rearrange
from torch import nn
#from einops import rearrange
import gin
import numpy as np
from .rotary_embedding import RotaryEmbedding

from typing import Dict, Tuple, Optional, List


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        num_channels: int,
        max_positions: int,
        factor: float,
        endpoint: bool = False,
        rearrange: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.factor = factor
        self.rearrange = (Rearrange("b (f c) -> b (c f)", f=2)
                          if rearrange else nn.Identity())

    def forward(self, x: torch.Tensor):
        x = x.view(-1)
        x = x * self.factor
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            device=x.device,
        ).float()
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return self.rearrange(x)


def chunk_wise_causal_mask(seq_len: int, chunk_size: int):
    # Initialize a full attention mask (everything allowed)
    mask = torch.zeros(seq_len, seq_len)

    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)

        # Allow full attention within the chunk
        mask[i:end, i:end] = 1  # Each chunk can fully attend to itself

        # Allow attention to all previous tokens
        mask[i:end, :i] = 1  # Attend to past chunks

    return 1 - mask  # Convert to mask format (1 = masked, 0 = allowed)


#@torch.jit.interface
#class ModuleInterface(torch.nn.Module):
#
#    def get_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
#        pass


class CacheModule(nn.Module):

    def __init__(self, max_cache_size: int = 0):
        super().__init__()
        self.max_cache_size = max_cache_size
        self.register_buffer('k_cache', torch.zeros(1))
        self.register_buffer('v_cache', torch.zeros(1))

    @torch.jit.export
    def get_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache, self.v_cache

    @torch.jit.export
    def set_cache(self, k: torch.Tensor, v: torch.Tensor):
        self.k_cache = k
        self.v_cache = v


class MHAttention(nn.Module):

    def __init__(self,
                 is_causal: bool = False,
                 dropout_level: float = 0.0,
                 n_heads: int = 4,
                 max_cache_size: int = 0,
                 rotary_emb: nn.Module = None,
                 embed_dim: int = 256,
                 min_chunk_size: int = 1,
                 max_num_cache=16,
                 max_batch_size=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

        if max_cache_size > 0:
            self.register_buffer('last_k', None)
            self.register_buffer('last_v', None)

            k_cache = torch.zeros((max_batch_size, max_num_cache, n_heads,
                                max_cache_size, embed_dim // n_heads))

            v_cache = torch.zeros((max_batch_size, max_num_cache, n_heads,
                                max_cache_size, embed_dim // n_heads))
            self.register_buffer('k_cache', k_cache)
            self.register_buffer('v_cache', v_cache)

        #self.cache = nn.ModuleList(
        #    [CacheModule(max_cache_size) for _ in range(max_num_cache)])

        self.rotary_emb = rotary_emb
        self.max_cache_size = max_cache_size
        self.min_chunk_size = min_chunk_size

        self.rearrange_heads1 = Rearrange("bs n (h d) -> bs h n d",
                                          h=self.n_heads)

        self.rearrange_heads2 = Rearrange("bs h n d -> bs n (h d)",
                                          h=self.n_heads)

    def get_buffers(self, i: int):
        k_cache, v_cache = self.k_cache[:, i], self.v_cache[:, i]
        return k_cache, v_cache

    def set_buffers(self, k, v, i: int):
        self.k_cache[:k.shape[0], i] = k
        self.v_cache[:k.shape[0], i] = v

    def roll_cache(self, roll_size: int, cache_index: int):
        k_cache, v_cache = self.get_buffers(cache_index)

        if roll_size < self.min_chunk_size:
            print("warming - roll size is smaller than min chunk size")

        k_cache = torch.cat(
            [k_cache[:self.last_k.shape[0]], self.last_k[:, :, :roll_size]],
            dim=2)
        v_cache = torch.cat(
            [v_cache[:self.last_k.shape[0]], self.last_v[:, :, :roll_size]],
            dim=2)

        if k_cache.shape[2] > self.max_cache_size:
            k_cache = k_cache[:, :, -self.max_cache_size:]
            v_cache = v_cache[:, :, -self.max_cache_size:]

        self.set_buffers(k_cache, v_cache, cache_index)

    def forward(self, q, k, v, cache_index: int):
        q, k, v = [self.rearrange_heads1(x) for x in [q, k, v]]

        if self.max_cache_size > 0:
            k_cache, v_cache = self.get_buffers(cache_index)
            if len(k_cache.shape) > 1:
                full_k = torch.cat([k_cache[:k.shape[0]], k], dim=2)
                full_v = torch.cat([v_cache[:k.shape[0]], v], dim=2)
                full_k = full_k[:, :, -self.max_cache_size:]
                full_v = full_v[:, :, -self.max_cache_size:]
            else:
                full_k = k
                full_v = v

            self.last_k = k
            self.last_v = v

        else:
            full_k = k
            full_v = v

        if self.is_causal:
            attn_mask = chunk_wise_causal_mask(full_k.shape[2],
                                               self.min_chunk_size)
            attn_mask = attn_mask[-q.shape[2]:]
            attn_mask = attn_mask.masked_fill(attn_mask == 1,
                                              float('-inf')).to(k)
        else:
            attn_mask = None

        if self.rotary_emb is not None:
            q, full_k = self.rotary_emb.rotate_queries_with_cached_keys(
                q, full_k)

        out = nn.functional.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=self.dropout_level if self.training else 0.)

        #out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)

        out = self.rearrange_heads2(out)
        return out


class SelfAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 is_causal=True,
                 dropout_level=0.0,
                 n_heads=8,
                 rotary_emb=None,
                 max_cache_size=0,
                 min_chunk_size=1):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal,
                               dropout_level,
                               n_heads,
                               rotary_emb=rotary_emb,
                               embed_dim=embed_dim,
                               max_cache_size=max_cache_size,
                               min_chunk_size=min_chunk_size)
        self.rotary_emb = rotary_emb

    def roll_cache(self, roll_size: int, cache_index: int):
        self.mha.roll_cache(roll_size, cache_index=cache_index)

    def forward(self, x, cache_index: int):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v, cache_index)


@gin.configurable
class CrossAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 is_causal=False,
                 dropout_level=0.,
                 n_heads=8,
                 rotary_emb=None,
                 max_cache_size=0,
                 min_chunk_size=1):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal,
                               dropout_level,
                               n_heads,
                               rotary_emb=rotary_emb,
                               max_cache_size=max_cache_size,
                               min_chunk_size=min_chunk_size)
        self.rotary_emb = rotary_emb

    def roll_cache(self, roll_size: int, cache_index: int):
        self.mha.roll_cache(roll_size, cache_index=cache_index)

    def forward(self, x, y, cache_index: int):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q, k, v, cache_index)


class MLP(nn.Module):

    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSepConv(nn.Module):

    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv1d(embed_dim,
                      mlp_multiplier * embed_dim,
                      kernel_size=1,
                      padding="same"),
            nn.Conv1d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=1,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv1d(mlp_multiplier * embed_dim,
                      embed_dim,
                      kernel_size=1,
                      padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        layer = Rearrange("b t c -> b c t")
        #x = rearrange(x, "b t c -> b c t")
        x = layer(x)
        x = self.mlp(x)
        layer = Rearrange("b c t -> b t c")
        x = layer(x)
        #x = rearrange(x, "b c t -> b t c")
        return x


class DoubleIdentity(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def roll_cache(self, size: int, cache_index: int):
        pass

    def forward(self, input: torch.Tensor, inputb: torch.Tensor,
                cache_index: int) -> torch.Tensor:
        return input


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        tcond_dim: int,
        use_crossattn: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        mlp_class,
        rotary_emb=None,
        max_cache_size: int = 0,
        min_chunk_size=1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.tcond_dim = tcond_dim
        self.self_attention = SelfAttention(embed_dim,
                                            is_causal,
                                            dropout_level,
                                            n_heads=embed_dim // 64,
                                            rotary_emb=rotary_emb,
                                            max_cache_size=max_cache_size,
                                            min_chunk_size=min_chunk_size)
        self.use_ca = use_crossattn
        if use_crossattn:
            self.cross_attention = CrossAttention(
                embed_dim,
                is_causal=False,
                dropout_level=0.,
                n_heads=embed_dim // 64,
                rotary_emb=rotary_emb,
                max_cache_size=max_cache_size,
                min_chunk_size=min_chunk_size)
            self.norm4 = nn.LayerNorm(embed_dim)
        else:
            self.cross_attention = DoubleIdentity()
            self.norm4 = nn.Identity()

        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        if self.cond_dim > 0:
            self.linear = nn.Linear(cond_dim, 2 * embed_dim)

        if self.tcond_dim > 0 and not self.use_ca:
            self.norm0 = nn.LayerNorm(embed_dim)
            self.tcond_linear = nn.Linear(tcond_dim, 2 * embed_dim)
        else:
            self.norm0 = nn.Identity()
            self.tcond_linear = nn.Identity()

    def roll_cache(self, size: int, cache_index: int = 0):
        self.self_attention.roll_cache(size, cache_index)
        if self.use_ca:
            self.cross_attention.roll_cache(size, cache_index)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor],
                tcond: Optional[torch.Tensor],
                cache_index: int) -> torch.Tensor:

        # AdaLN tcond
        if self.tcond_dim > 0 and not self.use_ca:
            x = self.norm0(x)
            assert tcond is not None
            alpha, beta = self.tcond_linear(tcond).chunk(2, dim=-1)
            x = x * (1 + alpha) + beta

        x = self.self_attention(self.norm1(x), cache_index=cache_index) + x

        # AdaLN cond
        x = self.norm2(x)
        if self.cond_dim > 0:
            assert cond is not None
            alpha, beta = self.linear(cond).chunk(2, dim=-1)
            x = x * (1 + alpha.unsqueeze(1)) + beta.unsqueeze(1)

        # Cross-Attention if time conditioning is activated and CA is used
        if self.tcond_dim > 0 and self.use_ca:
            assert tcond is not None
            x = self.cross_attention(
                self.norm4(x), tcond, cache_index=cache_index) + x

        # Final layer

        x = self.mlp(self.norm3(x)) + x
        return x


class DenoiserTransBlock(nn.Module):

    def __init__(self,
                 n_channels: int = 64,
                 seq_len: int = 32,
                 mlp_multiplier: int = 4,
                 noise_embed_dims: int = 64,
                 embed_dim: int = 256,
                 cond_dim: int = 128,
                 tcond_dim: int = 0,
                 dropout: float = 0.1,
                 n_layers: int = 4,
                 is_causal: bool = True,
                 tcond_mode: str = "cross_attention",
                 temporal_noise_dim: int = 0,
                 pos_emb_type: str = "learnable",
                 max_cache_size: int = 0,
                 min_chunk_size: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        self.patchify_and_embed = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(n_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.use_crossattn = (tcond_dim > 0
                              and tcond_mode == "cross_attention")

        tcond_dim = tcond_dim + temporal_noise_dim

        if tcond_dim > 0:
            if self.use_crossattn:
                self.patchify_and_embed_tcond = nn.Sequential(
                    Rearrange("b c t -> b t c"),
                    nn.Linear(tcond_dim, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                )
                self.pos_embed_ca = nn.Embedding(seq_len, self.embed_dim)
                self.register_buffer("precomputed_pos_enc_ca",
                                     torch.arange(0, seq_len).long())
                tcond_dim = embed_dim
            else:
                self.patchify_and_embed_tcond = nn.Sequential(
                    Rearrange("b c t -> b t c"),
                    nn.Linear(tcond_dim, tcond_dim),
                    nn.LayerNorm(tcond_dim),
                )
                self.pos_embed_ca = nn.Identity()

        else:
            self.patchify_and_embed_tcond = nn.Identity()

        self.rearrange2 = Rearrange("b t c -> b c t", )

        if pos_emb_type == "learnable":
            self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        elif pos_emb_type == "rotary":
            self.rotary_emb = RotaryEmbedding(32)
            self.pos_embed = None
        elif pos_emb_type == "none":
            self.pos_embed = None

        precomputed_pos_enc = torch.arange(0, seq_len).long()
        self.register_buffer("precomputed_pos_enc", precomputed_pos_enc)
        self.register_buffer("precomputed_pos_enc_ca", precomputed_pos_enc)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=self.embed_dim,
                mlp_multiplier=self.mlp_multiplier,
                cond_dim=cond_dim,
                tcond_dim=tcond_dim,
                use_crossattn=self.use_crossattn,
                is_causal=is_causal,
                dropout_level=self.dropout,
                mlp_class=MLP,
                rotary_emb=None
                if pos_emb_type != "rotary" else self.rotary_emb,
                max_cache_size=max_cache_size,
                min_chunk_size=min_chunk_size,
            ) for _ in range(self.n_layers)
        ])
        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, n_channels),
                                      self.rearrange2)

    def roll_cache(self, size: int, cache_index: int):
        for block in self.decoder_blocks:
            block.roll_cache(size, cache_index)

    def forward(self, x: torch.Tensor, features: Optional[torch.Tensor],
                tcond: Optional[torch.Tensor],
                temporal_noise_level: Optional[torch.Tensor],
                cache_index: int):

        x = self.patchify_and_embed(x)

        if self.pos_embed is not None:
            pos_enc = self.pos_embed(
                self.precomputed_pos_enc[:x.size(1)]).expand(
                    x.size(0), x.size(1), -1)
            x = x + pos_enc

        if tcond is not None:
            if temporal_noise_level is not None:
                tcond = torch.cat([tcond, temporal_noise_level], dim=1)

            tcond = self.patchify_and_embed_tcond(tcond)

            if self.use_crossattn:
                pos_enc_ca = self.precomputed_pos_enc_ca[:tcond.
                                                         size(1)].expand(
                                                             tcond.size(0), -1)
                tcond = tcond + self.pos_embed_ca(pos_enc_ca)

        elif tcond is None and temporal_noise_level is not None:
            tcond = temporal_noise_level
            tcond = self.patchify_and_embed_tcond(tcond)
        else:
            tcond = None

        for block in self.decoder_blocks:

            x = block(x, cond=features, tcond=tcond, cache_index=cache_index)

        return self.out_proj(x)


@gin.configurable
class Denoiser(nn.Module):

    def __init__(
        self,
        n_channels: int,
        seq_len: int = 32,
        embed_dim: int = 256,
        cond_dim: int = 64,
        tcond_dim: int = 0,
        noise_embed_dims: int = 128,
        n_layers: int = 6,
        mlp_multiplier: int = 2,
        dropout: float = 0.1,
        causal: bool = False,
        tcond_mode: str = "cross_attention",
        temporal_noise: bool = False,
        pos_emb_type="learnable",
        max_cache_size: int = 0,
        min_chunk_size: int = 1,
    ):
        super().__init__()
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.temporal_noise = temporal_noise

        self.fourier_feats = PositionalEmbedding(num_channels=noise_embed_dims,
                                                 max_positions=10_000,
                                                 factor=100.0)

        if self.temporal_noise:
            embedding_in_dims = cond_dim
        else:
            embedding_in_dims = noise_embed_dims + cond_dim

        self.embedding = nn.Sequential(
            nn.Linear(embedding_in_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.denoiser_trans_block = DenoiserTransBlock(
            n_channels=n_channels,
            seq_len=seq_len,
            mlp_multiplier=mlp_multiplier,
            embed_dim=embed_dim,
            dropout=dropout,
            n_layers=n_layers,
            cond_dim=0 if
            (self.temporal_noise and cond_dim == 0) else self.embed_dim,
            tcond_dim=tcond_dim,
            is_causal=causal,
            tcond_mode=tcond_mode,
            temporal_noise_dim=noise_embed_dims if self.temporal_noise else 0,
            pos_emb_type=pos_emb_type,
            max_cache_size=max_cache_size,
            min_chunk_size=min_chunk_size)

        self.rearrange_forward1 = Rearrange("b t-> (b t)")
        self.rearrange_forward2 = Rearrange("(b t) c -> b c t", t=seq_len)

    @property
    def name(self):
        return "transformer"

    def roll_cache(self, size: int, cache_index: int):
        self.denoiser_trans_block.roll_cache(size, cache_index)

    def forward(self,
                x,
                time: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                time_cond: Optional[torch.Tensor] = None,
                cache_index: int = 0) -> torch.Tensor:

        if self.temporal_noise:
            if time.shape[-1] == 1 or len(time.shape) == 1:
                time = time.reshape(-1, 1).repeat(1, x.shape[-1])

            if len(time.shape) > 2:
                time = time.squeeze(1)
            seq_len = x.shape[-1]

            #time = self.rearrange_forward1(time)
            #
            #time = einops.rearrange(time, "b t -> (b t)")
            #temporal_noise_level = self.fourier_feats(time)
            #layer = Rearrange("(b t) c -> b c t", t=seq_len)
            #temporal_noise_level = einops.rearrange(temporal_noise_level,
            #                                 "(b t) c -> b c t",
            #                                 t=seq_len)

            time = time.reshape(-1)  # Equivalent to "b t -> (b t)"
            temporal_noise_level = self.fourier_feats(time)
            temporal_noise_level = temporal_noise_level.view(
                -1, seq_len, temporal_noise_level.shape[-1]).permute(0, 2, 1)

            # temporal_noise_level = layer(temporal_noise_level)
            if cond is not None:
                features = self.embedding(cond)
            else:
                features = cond

        else:
            if len(time.shape) > 1 and time.shape[-1] != 1:
                raise ValueError(
                    "Without temporal_noise activated, time should be a 1D tensor"
                )
            time = time.reshape(-1)
            noise_level = self.fourier_feats(time)
            if cond is not None:
                embedding_in = torch.cat([noise_level, cond], dim=-1)
            else:
                embedding_in = noise_level
            temporal_noise_level = None

            features = self.embedding(embedding_in)

        x = self.denoiser_trans_block(
            x,
            features=features,
            tcond=time_cond,
            temporal_noise_level=temporal_noise_level,
            cache_index=cache_index)

        return x


if __name__ == "__main__":
    denoiser = Denoiser(n_channels=16,
                        tcond_dim=24,
                        cond_dim=32,
                        tcond_mode="cross_attention",
                        temporal_noise=False)

    tcond = torch.randn((16, 24, 32))
    x = torch.randn((16, 16, 32))
    cond = torch.randn((16, 32))

    time = torch.randn(16, 1)

    print(denoiser(x, time, cond, tcond).shape)

    denoiser = Denoiser(n_channels=16,
                        tcond_dim=24,
                        cond_dim=32,
                        tcond_mode="adaln")

    tcond = torch.randn((16, 24, 32))
    x = torch.randn((16, 16, 32))
    cond = torch.randn((16, 32))

    time = torch.randn(16)

    print(denoiser(x, time, cond, tcond).shape)