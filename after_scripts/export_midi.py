import torch.nn as nn
import nn_tilde
import torch

import argparse
import os
from after.diffusion import RectifiedFlow

torch.set_grad_enabled(False)

import gin
import cached_conv as cc
import numpy as np
from absl import flags, app

cc.use_cached_conv(True)
parser = argparse.ArgumentParser()

FLAGS = flags.FLAGS
# Flags definition
flags.DEFINE_string("model_path",
                    default="./after_runs/test",
                    help="Name of the experiment folder")
flags.DEFINE_integer("step", default=None, help="Step number of checkpoint")
flags.DEFINE_string("emb_model_path",
                    default="./pretrained/test.ts",
                    help="Path to audio codec")
flags.DEFINE_integer("chunk_size", default=4, help="Chunk size")
flags.DEFINE_integer("max_cache_size", default=128, help="Max cache size")
flags.DEFINE_integer("n_poly", default=8, help="Number of polyphonic voices")
flags.DEFINE_integer("train_latent_map",
                     default=False,
                     help="Train a 2D latent map for max4Live Device")


def main(argv):
    # Parse model folder
    folder = FLAGS.model_path

    if FLAGS.step is None:
        files = os.listdir(folder)
        files = [f for f in files if f.startswith("checkpoint")]
        steps = [f.split("_")[-2].replace("checkpoint", "") for f in files]
        step = max([int(s) for s in steps])
        checkpoint_file = "checkpoint" + str(step) + "_EMA.pt"
    else:
        checkpoint_file = "checkpoint" + str(FLAGS.step) + "_EMA.pt"

    print("Using checkpoint at step : ", checkpoint_file)

    checkpoint_path = os.path.join(folder, checkpoint_file)
    config = folder + "/config.gin"

    out_name = os.path.join(folder,
                            "after.midi." + folder.split("/")[-1] + ".ts")
    # Parse config
    gin.parse_config_file(config)
    SR = gin.query_parameter("%SR")

    with gin.unlock_config():
        try:
            gin.bind_parameter("transformerv2.MHAttention.max_cache_size",
                               gin.query_parameter("%LOCAL_ATTENTION_SIZE"))
        except:
            gin.bind_parameter("transformer.Denoiser.max_cache_size",
                               gin.query_parameter("%N_SIGNAL"))

    # Instantiate model
    blender = RectifiedFlow()

    # Load checkpoints
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    blender.load_state_dict(state_dict, strict=False)

    # Emb model
    # Send to device
    blender = blender.eval()

    # Get some parameters
    n_signal = gin.query_parameter('%N_SIGNAL')
    n_signal_timbre = gin.query_parameter('%N_SIGNAL')
    zt_channels = gin.query_parameter("%ZT_CHANNELS")
    ae_latents = gin.query_parameter("%IN_SIZE")

    class Streamer(nn_tilde.Module):

        def __init__(self) -> None:
            super().__init__()

            self.net = blender.net
            self.encoder = blender.encoder
            self.encoder_time = blender.encoder_time

            self.n_signal = n_signal
            self.n_signal_timbre = n_signal_timbre
            self.chunk_size = FLAGS.chunk_size
            self.n_poly = FLAGS.n_poly
            self.zt_channels = zt_channels
            self.ae_latents = ae_latents
            self.emb_model_timbre = torch.jit.load(FLAGS.emb_model_path).eval()

            self.drop_value = blender.drop_value

            # Get the ae ratio
            dummy = torch.zeros(1, 1, 4 * 4096)
            z = self.emb_model_timbre.encode(dummy)
            self.ae_ratio = 4 * 4096 // z.shape[-1]

            self.sr = gin.query_parameter("%SR")
            self.zt_buffer = self.n_signal_timbre * self.ae_ratio

            ## ATTRIBUTES ##
            self.register_attribute("nb_steps", 1)
            self.register_attribute("guidance", 1.)
            self.register_attribute("guidance_timbre", 1.)
            self.register_attribute("guidance_structure", 1.)
            self.register_attribute("learn_zsem", False)

            ## BUFFERS ##
            self.register_buffer(
                "previous_timbre",
                torch.zeros(4, self.ae_latents, self.n_signal_timbre))

            self.register_buffer("last_zsem", torch.zeros(4, self.zt_channels))

            ## METHODS ##

            input_labels = [
                f"(signal) Input {l} {i}" for i in range(self.n_poly)
                for l in ["pitch", "velocity"]
            ]
            self.register_method(
                "forward",
                in_channels=self.n_poly * 2 + 1,
                in_ratio=1,
                out_channels=1,
                out_ratio=1,
                input_labels=input_labels + [f"(signal) Input timbre"],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "timbre",
                in_channels=1,
                in_ratio=1,
                out_channels=self.zt_channels,
                out_ratio=self.ae_ratio,
                input_labels=[
                    f"(signal) Input timbre",
                ],
                output_labels=[
                    f"(signal) Output timbre {i}" for i in range(zt_channels)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "generate",
                in_channels=self.n_poly * 2 + zt_channels,
                in_ratio=self.ae_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=input_labels +
                [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "diffuse",
                in_channels=self.n_poly * 2 + zt_channels,
                in_ratio=self.ae_ratio,
                out_channels=self.ae_latents,
                out_ratio=self.ae_ratio,
                input_labels=input_labels +
                [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=[
                    f"(signal) Latent output {i}"
                    for i in range(self.ae_latents)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "decode",
                in_channels=self.ae_latents,
                in_ratio=self.ae_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Latent input {i}"
                    for i in range(self.ae_latents)
                ],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

        @torch.jit.export
        def get_learn_zsem(self) -> bool:
            return self.learn_zsem[0]

        @torch.jit.export
        def set_learn_zsem(self, learn_zsem: bool) -> int:
            self.learn_zsem = (learn_zsem, )
            return 0

        @torch.jit.export
        def get_guidance(self) -> float:
            return self.guidance[0]

        @torch.jit.export
        def set_guidance(self, guidance: float) -> int:
            self.guidance = (guidance, )
            return 0

        @torch.jit.export
        def get_guidance_timbre(self) -> float:
            return self.guidance_timbre[0]

        @torch.jit.export
        def set_guidance_timbre(self, guidance_timbre: float) -> int:
            self.guidance_timbre = (guidance_timbre, )
            return 0

        @torch.jit.export
        def get_guidance_structure(self) -> float:
            return self.guidance_structure[0]

        @torch.jit.export
        def set_guidance_structure(self, guidance_structure: float) -> int:
            self.guidance_structure = (guidance_structure, )
            return 0

        @torch.jit.export
        def get_nb_steps(self) -> int:
            return self.nb_steps[0]

        @torch.jit.export
        def set_nb_steps(self, nb_steps: int) -> int:
            self.nb_steps = (nb_steps, )
            return 0

        def model_forward(self, x: torch.Tensor, time: torch.Tensor,
                          cond: torch.Tensor, time_cond: torch.Tensor,
                          cache_index: int) -> torch.Tensor:

            guidance_timbre = self.guidance_timbre[0]
            guidance_structure = self.guidance_structure[0]

            full_time = time.repeat(3, 1, 1)
            full_x = x.repeat(3, 1, 1)

            full_cond = torch.cat([
                cond,
                cond,
                self.drop_value * torch.ones_like(cond),
            ])

            full_time_cond = torch.cat([
                time_cond,
                self.drop_value * torch.ones_like(time_cond),
                self.drop_value * torch.ones_like(time_cond),
            ])

            dx = self.net(full_x,
                          time=full_time,
                          cond=full_cond,
                          time_cond=full_time_cond,
                          cache_index=cache_index)

            dx_full, dx_cond, dx_none = torch.chunk(dx, 3, dim=0)

            total_guidance = 0.5 * (guidance_structure + guidance_timbre)

            guidance_cond_factor = guidance_structure / (max(
                guidance_timbre, 0.1))

            dx = dx_none + total_guidance * (dx_cond + guidance_cond_factor *
                                             (dx_full - dx_cond) - dx_none)

            return dx

        def sample(self, x_last: torch.Tensor, cond: torch.Tensor,
                   time_cond: torch.Tensor):

            x = x_last
            t = torch.linspace(0, 1, self.nb_steps[0] + 1)
            dt = 1 / self.nb_steps[0]

            for i, t_value in enumerate(t[:-1]):

                dt = dt

                x = x + self.model_forward(x=x,
                                           time=t_value.repeat(
                                               x.shape[0], 1, x.shape[-1]),
                                           cond=cond,
                                           time_cond=time_cond,
                                           cache_index=i) * dt

                self.net.roll_cache(x.shape[-1], i)
            return x

        @torch.jit.export
        def timbre(self, x) -> torch.Tensor:
            x = self.emb_model_timbre.encode(x)

            self.previous_timbre[:x.shape[0]] = torch.cat(
                (self.previous_timbre[:x.shape[0]], x), -1)[..., x.shape[-1]:]

            zsem = self.encoder.forward_stream(
                self.previous_timbre[:x.shape[0]])

            zsem = zsem.unsqueeze(-1).repeat((1, 1, x.shape[-1]))
            return zsem

        @torch.jit.export
        def diffuse(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            zsem = x[:, -self.zt_channels:].mean(-1)

            # Get the notes
            notes = x[:, :2 * self.n_poly]
            time_cond = torch.zeros((1, 128, x.shape[-1]))

            for i in range(self.n_poly):
                for j in range(x.shape[-1]):
                    if notes[0, 2 * i + 1, j] > 0:
                        time_cond[:, notes[:, 2 * i].long(),
                                  j] = notes[:, 2 * i + 1, j] / 128

            # Generate
            x = torch.randn(n, self.ae_latents, x.shape[-1])

            x = self.sample(x[:1], time_cond=time_cond[:1], cond=zsem[:1])

            if n > 1:
                x = x.repeat(n, 1, 1)
            return x

        @torch.jit.export
        def decode(self, x: torch.Tensor) -> torch.Tensor:
            audio = self.emb_model_timbre.decode(x)
            return audio

        @torch.jit.export
        def generate(self, x: torch.Tensor) -> torch.Tensor:
            z = self.diffuse(x)
            audio = self.decode(z)
            return audio

    ####
    streamer = Streamer()
    dummmy = torch.randn(1, FLAGS.n_poly * 2 + zt_channels, FLAGS.chunk_size)
    out = streamer.diffuse(dummmy)

    streamer.export_to_ts(out_name)

    print("Bravo - Export successful")


if __name__ == "__main__":
    app.run(main)
