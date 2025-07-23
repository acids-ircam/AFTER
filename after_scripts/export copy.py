import torch.nn as nn
import nn_tilde
import torch

import argparse
import os
from after.diffusion import RectifiedFlow
from after.dataset import CombinedDataset
from after.diffusion.latent_plot import prepare_training, train_autoencoder, generate_plot

torch.set_grad_enabled(False)

import gin
import cached_conv as cc
from absl import flags, app

cc.use_cached_conv(True)
parser = argparse.ArgumentParser()

FLAGS = flags.FLAGS
# Flags definition
flags.DEFINE_string("model_path",
                    default="./after_runs/test",
                    help="Name of the experiment folder")
flags.DEFINE_integer(
    "step",
    default=None,
    help="Step number of checkpoint - use None to use the last checkpoint")
flags.DEFINE_string("emb_model_path",
                    default="./pretrained/test.ts",
                    help="Path to encoder model")
flags.DEFINE_integer("chunk_size", default=4, help="Chunk size")
flags.DEFINE_bool("latent_project",
                  default=True,
                  help="Create latent map embedding plot")
flags.DEFINE_float(
    "latent_range",
    default=1.0,
    help="Scale the latent space visualisation to [-latent_range, latent_range]"
)


class DummyIdentity(nn.Module):
    """Dummy identity model for compatibility"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()


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

    # Parse config
    gin.parse_config_file(config)
    SR = gin.query_parameter("%SR")

    with gin.unlock_config():
        with gin.unlock_config():
            try:
                cache_denoiserv2 = gin.query_parameter("%LOCAL_ATTENTION_SIZE")
                gin.bind_parameter("transformerv2.MHAttention.max_cache_size",
                                   cache_denoiserv2)
            except:
                cache_denoiserv1 = gin.query_parameter("%N_SIGNAL")
                gin.bind_parameter("transformer.Denoiser.max_cache_size",
                                   cache_denoiserv1)

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
    zs_channels = gin.query_parameter("%ZS_CHANNELS")
    ae_latents = gin.query_parameter("%IN_SIZE")

    ### GENERATE EMBEDDING PLOT ###
    ### GENERATE EMBEDDING PLOT ###
    if FLAGS.latent_project:

        try:
            path_dict = gin.query_parameter("utils.get_datasets.path_dict")
            dataset = CombinedDataset(path_dict=path_dict,
                                      keys=["z", "metadata"])

            embeddings, labels = prepare_training(blender.encoder,
                                                  dataset,
                                                  num_examples=3000)

            embeddings = embeddings / (FLAGS.latent_range)
            torch.set_grad_enabled(True)
            if zt_channels > 2:
                project_model = train_autoencoder(embeddings,
                                                  num_steps=50000,
                                                  batch_size=8,
                                                  lr=1e-3,
                                                  device="cpu")

                compressed_embeddings = project_model.encoder(
                    torch.tensor(embeddings,
                                 dtype=torch.float32)).detach().numpy()

            fig, _ = generate_plot(compressed_embeddings,
                                   labels,
                                   use_blur=True,
                                   bins=500,
                                   sigma=12,
                                   gamma=0.5,
                                   brightness_scale=1.)
            torch.set_grad_enabled(False)
        except Exception as e:
            print("Could not load dataset for embedding plot.")
            print("Error : ", e)
            print("Us --nolatent_project to disable latent projection.")
            exit()
    else:
        project_model = DummyIdentity()

    class Streamer(nn_tilde.Module):

        def __init__(self) -> None:
            super().__init__()

            self.net = blender.net
            self.encoder = blender.encoder
            self.encoder_time = blender.encoder_time

            self.n_signal = n_signal
            self.n_signal_timbre = n_signal_timbre
            self.chunk_size = FLAGS.chunk_size
            self.zs_channels = zs_channels
            self.zt_channels = zt_channels
            self.ae_latents = ae_latents
            self.emb_model_structure = torch.jit.load(
                FLAGS.emb_model_path).eval()

            self.project_model = project_model

            # self.emb_model_timbre = torch.jit.load(FLAGS.emb_model_path).eval()

            self.drop_value = blender.drop_value

            self.latent_range = FLAGS.latent_range

            # Get the ae ratio
            dummy = torch.zeros(1, 1, 4 * 4096)
            z = self.emb_model_structure.encode(dummy)
            self.ae_ratio = 4 * 4096 // z.shape[-1]

            self.sr = gin.query_parameter("%SR")
            self.zt_buffer = self.n_signal_timbre * self.ae_ratio

            ## ATTRIBUTES ##
            self.register_attribute("nb_steps", 1)
            self.register_attribute("guidance_timbre", 1.)
            self.register_attribute("guidance_structure", 1.)

            ## BUFFERS ##
            self.register_buffer(
                "previous_timbre",
                torch.zeros(4, self.ae_latents, self.n_signal_timbre))

            ## METHODS ##
            # self.register_method(
            #     "forward",
            #     in_channels=2,
            #     in_ratio=1,
            #     out_channels=1,
            #     out_ratio=1,
            #     input_labels=[
            #         f"(signal) Input structure",
            #         f"(signal) Input timbre",
            #     ],
            #     output_labels=[f"(signal) Audio output"],
            #     test_buffer_size=self.chunk_size * self.ae_ratio,
            # )

            self.register_method(
                "structure",
                in_channels=1,
                in_ratio=1,
                out_channels=self.zs_channels,
                out_ratio=self.ae_ratio,
                input_labels=[
                    f"(signal) Input structure",
                ],
                output_labels=[
                    f"(signal) Output structure {i}"
                    for i in range(zs_channels)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            # self.register_method(
            #     "timbre",
            #     in_channels=1,
            #     in_ratio=1,
            #     out_channels=self.zt_channels,
            #     out_ratio=self.ae_ratio,
            #     input_labels=[
            #         f"(signal) Input timbre",
            #     ],
            #     output_labels=[
            #         f"(signal) Output timbre {i}" for i in range(zt_channels)
            #     ],
            #     test_buffer_size=self.chunk_size * self.ae_ratio,
            # )

            self.register_method(
                "diffuse",
                in_channels=zt_channels + zs_channels,
                in_ratio=self.ae_ratio,
                out_channels=self.ae_latents,
                out_ratio=self.ae_ratio,
                input_labels=[
                    f"(signal) Input structure {i}"
                    for i in range(self.zs_channels)
                ] + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ],
                output_labels=[
                    f"(signal) Latent output {i}"
                    for i in range(self.ae_latents)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "generate",
                in_channels=zt_channels + zs_channels,
                in_ratio=self.ae_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Input structure {i}" for i in range(zs_channels)
                ] + [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "generate_timbre",
                in_channels=zt_channels + 1,
                in_ratio=1,
                out_channels=1,
                out_ratio=1,
                input_labels=[f"(signal) audio structure"] + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ],
                output_labels=[f"(signal) audio out"],
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

            self.register_method(
                "latent2map",
                in_channels=2
                if not FLAGS.latent_project else self.zt_channels,
                in_ratio=1,
                out_channels=2,
                out_ratio=1,
                input_labels=[
                    f"(signal_{i}) Full Latent" for i in range(
                        2 if not FLAGS.latent_project else self.zt_channels)
                ],
                output_labels=[
                    f"(signal) 2D Latent 1", f"(signal) 2D Latent 2"
                ],
                test_buffer_size=2048,
            )

            self.register_method(
                "map2latent",
                in_channels=2,
                in_ratio=1,
                out_channels=2
                if not FLAGS.latent_project else self.zt_channels,
                out_ratio=1,
                output_labels=[
                    f"(signal_{i}) Full Latent" for i in range(
                        2 if not FLAGS.latent_project else self.zt_channels)
                ],
                input_labels=[
                    f"(signal) 2D Latent 1", f"(signal) 2D Latent 2"
                ],
                test_buffer_size=2048,
            )

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
                self.drop_value * torch.ones_like(cond),
                self.drop_value * torch.ones_like(cond),
            ])

            full_time_cond = torch.cat([
                time_cond,
                time_cond,
                self.drop_value * torch.ones_like(time_cond),
            ])

            dx = self.net(full_x,
                          time=full_time,
                          cond=full_cond,
                          time_cond=full_time_cond,
                          cache_index=cache_index)

            dx_full, dx_time_cond, dx_none = torch.chunk(dx, 3, dim=0)

            total_guidance = 0.5 * (guidance_structure + guidance_timbre)

            guidance_cond_factor = guidance_timbre / (max(
                guidance_structure, 0.1))

            dx = dx_none + total_guidance * (
                dx_time_cond + guidance_cond_factor *
                (dx_full - dx_time_cond) - dx_none)

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

        # @torch.jit.export
        # def timbre(self, x) -> torch.Tensor:
        #     x = self.emb_model_timbre.encode(x)
        #     self.previous_timbre[:x.shape[0]] = torch.cat(
        #         (self.previous_timbre[:x.shape[0]], x), -1)[..., x.shape[-1]:]

        #     zsem = self.encoder.forward_stream(
        #         self.previous_timbre[:x.shape[0]])

        #     zsem = zsem / self.latent_range

        #     return zsem.unsqueeze(-1).repeat((1, 1, self.chunk_size))

        @torch.jit.export
        def structure(self, x) -> torch.Tensor:
            x = self.emb_model_structure.encode(x)
            x = self.encoder_time.forward(x)
            
            return x

        @torch.jit.export
        def diffuse(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            zsem = x[:, -self.zt_channels:].mean(-1)
            zsem = zsem * self.latent_range
            time_cond = x[:, :self.zs_channels]
            x = torch.randn(n, self.ae_latents, x.shape[-1])
            x = self.sample(x[:1], time_cond=time_cond[:1], cond=zsem[:1])

            if n > 1:
                x = x.repeat(n, 1, 1)
            return x

        @torch.jit.export
        def diffuse_timbre(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            zsem = x[:, 1:].mean(-1)
            zsem = zsem * self.latent_range

            audio = x[:, :1, :]
            print(audio.shape)
            time_cond = self.structure(audio)

            x = torch.randn(n, self.ae_latents, time_cond.shape[-1])
            x = self.sample(x[:1], time_cond=time_cond[:1], cond=zsem[:1])

            if n > 1:
                x = x.repeat(n, 1, 1)
            return x

        @torch.jit.export
        def decode(self, x: torch.Tensor) -> torch.Tensor:
            audio = self.emb_model_structure.decode(x)
            return audio

        @torch.jit.export
        def generate(self, x: torch.Tensor) -> torch.Tensor:
            z = self.diffuse(x)
            audio = self.decode(z)
            return audio

        @torch.jit.export
        def generate_timbre(self, x: torch.Tensor) -> torch.Tensor:
            z = self.diffuse_timbre(x)
            audio = self.decode(z)
            return audio

        # def forward(self, x: torch.Tensor) -> torch.Tensor:
        #     x_structure, x_timbre = x[:, :1], x[:, 1:]
        #     structure, timbre = self.structure(x_structure), self.timbre(
        #         x_timbre)
        #     x = torch.cat((structure, timbre), 1)
        #     z = self.diffuse(x)
        #     audio = self.decode(z)
        #     return audio

        @torch.jit.export
        def map2latent(self, x: torch.Tensor) -> torch.Tensor:
            tdim = x.shape[-1]
            mapvec = x.mean(-1)
            latents = self.project_model.decoder(mapvec)
            return latents.unsqueeze(-1).repeat((1, 1, tdim))

        @torch.jit.export
        def latent2map(self, x: torch.Tensor) -> torch.Tensor:
            tdim = x.shape[-1]
            latents = x.mean(-1)
            map = self.project_model.encoder(latents)
            return map.unsqueeze(-1).repeat((1, 1, tdim))

    ####

    streamer = Streamer()

    dummmy = torch.randn(1, zs_channels + zt_channels, FLAGS.chunk_size)
    out = streamer.diffuse(dummmy)
    out_name = os.path.join(folder,
                            "after.audio." + folder.split("/")[-1] + ".ts")

    streamer.export_to_ts(out_name)

    out_name_plot = os.path.join(
        folder, "after.audio." + folder.split("/")[-1] + ".png")

    if FLAGS.latent_project:
        fig.savefig(out_name_plot,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor=fig.get_facecolor(),
                    transparent=False)

    print("Bravo - Export successful")


if __name__ == "__main__":
    app.run(main)
