import torch.nn as nn
import nn_tilde
import torch

import argparse
import os
import pathlib
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
flags.DEFINE_bool("latent_project",
                  default=True,
                  help="Train a 2D latent map for max4Live Device")
flags.DEFINE_float(
    "latent_range",
    default=1.0,
    help="Scale the latent space visualisation to [-latent_range, latent_range]"
)
flags.DEFINE_string(
    "label_mode",
    default="dataset",
    help=
    "Mode for labeling data and color the map. Chose( between 'dataset' for multi dataset and 'file' when using meaningful file names."
)

flags.DEFINE_integer("num_steps",
                     default=20000,
                     help="Number of steps to train the AE map model")

flags.DEFINE_integer(
    "num_examples",
    default=None,
    help=
    "Number of sampled examples to build the map - defaults to full dataset")

flags.DEFINE_string(
    "ae_mode",
    default="linear",
    help=
    "Default organisation of the map (linear, spherical, lambert). Spherical and lambert are mapped on to a sphere and with (polar/lambert) coordinates respectively, linear is a simple 2D map."
)

flags.DEFINE_bool("reload_embeddings",
                  default=False,
                  help="Reload precomputed timbre embeddings if available")

flags.DEFINE_multi_string(
    "db_path",
    default=[],
    help=
    "Dataset path(s) for the latent map. Overrides the db_list from the config when provided."
)

flags.DEFINE_string(
    "db_folder",
    default=None,
    help=
    "Folder whose sub-directories are each an LMDB dataset. Merged with --db_path entries."
)


class DummyIdentity(nn.Module):
    """Dummy identity model for compatibility"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x: torch.Tensor):
        return x

    def forward_stream(self, x: torch.Tensor):
        return x

    def encode(self, x: torch.Tensor):
        return x

    def decode(self, x: torch.Tensor):
        return x


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
        try:
            gin.bind_parameter("transformerv2.DenoiserV2.streaming", True)
        except ValueError:
            try:
                cache_size = gin.query_parameter("%LOCAL_ATTENTION_SIZE")
                gin.bind_parameter("transformerv2.MHAttention.max_cache_size",
                                   cache_size)
            except ValueError:
                try:
                    n_signal = gin.query_parameter("%N_SIGNAL")
                    gin.bind_parameter("transformer.Denoiser.max_cache_size",
                                       n_signal)
                except ValueError:
                    pass

    # Resolve codec path
    emb_model_path = FLAGS.emb_model_path
    if emb_model_path == "./pretrained/test.ts" or emb_model_path is None:
        try:
            trained_path = gin.query_parameter(
                "diffusion.utils.get_datasets.emb_model_path")
        except ValueError:
            trained_path = None
        if not trained_path:
            raise RuntimeError(
                "No --emb_model_path provided and none saved in config.\n"
                "Re-run training or pass --emb_model_path.")
        stream_path = str(
            pathlib.Path(trained_path).with_stem(
                pathlib.Path(trained_path).stem + "_stream"))
        if os.path.exists(stream_path):
            emb_model_path = stream_path
            print(f"Using streaming codec: {emb_model_path}")
        else:
            raise FileNotFoundError(
                f"Streaming codec not found at '{stream_path}'.\n"
                f"Please provide one explicitly with --emb_model_path.")

    # Instantiate model
    blender = RectifiedFlow()

    # Load checkpoints
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    blender.load_state_dict(state_dict, strict=False)

    # Emb model
    # Send to device
    blender = blender.eval()

    # Get some parameters
    n_signal = n_signal_timbre = gin.query_parameter('%N_SIGNAL')
    zt_channels = gin.query_parameter("%ZT_CHANNELS")
    zs_channels = gin.query_parameter("%ZS_CHANNELS")
    ae_latents = gin.query_parameter("%IN_SIZE")
    model_chunk_size = gin.query_parameter("%ATTENTION_CHUNK_SIZE")

    ### GENERATE EMBEDDING PLOT ###
    if FLAGS.latent_project:

        if True:
            if FLAGS.db_path or FLAGS.db_folder:
                db_list = list(FLAGS.db_path)
                if FLAGS.db_folder is not None:
                    subdirs = sorted([
                        str(p)
                        for p in pathlib.Path(FLAGS.db_folder).iterdir()
                        if p.is_dir()
                    ])
                    db_list += subdirs
            else:
                try:
                    db_list = gin.query_parameter(
                        "diffusion.utils.get_datasets.db_list")
                except ValueError:
                    try:
                        path_dict_legacy = gin.query_parameter(
                            "utils.get_datasets.path_dict")
                        db_list = list(path_dict_legacy.keys())
                    except ValueError:
                        db_list = []
            path_dict = {k: {"path": k, "name": k} for k in db_list}
            dataset = CombinedDataset(path_dict=path_dict,
                                      keys=["z", "metadata"])

            tmp = os.path.join(FLAGS.model_path, "latent_embeddings.pt")

            if FLAGS.reload_embeddings and os.path.exists(tmp):
                embeddings, labels = torch.load(tmp)
            else:
                embeddings, labels = prepare_training(
                    encoder=blender.encoder,
                    post_encoder=blender.post_encoder,
                    dataset=dataset,
                    num_examples=FLAGS.num_examples,
                    mode=FLAGS.label_mode)

                torch.save((embeddings, labels), tmp)

            print("Labels for the map : ", set(labels))

            embeddings = embeddings / (FLAGS.latent_range)
            torch.set_grad_enabled(True)
            if zt_channels > 2:
                project_model = train_autoencoder(embeddings,
                                                  num_steps=FLAGS.num_steps,
                                                  batch_size=128,
                                                  lr=1e-4,
                                                  device="cpu",
                                                  val_split=0.05,
                                                  mode=FLAGS.ae_mode)

                compressed_embeddings = project_model.encode(
                    torch.tensor(embeddings,
                                 dtype=torch.float32)).detach().numpy()
            else:
                compressed_embeddings = embeddings
                project_model = DummyIdentity()

            fig, legend_fig = generate_plot(compressed_embeddings,
                                            labels,
                                            use_blur=True,
                                            bins=100,
                                            sigma=2,
                                            gamma=1.,
                                            brightness_scale=10.)
            torch.set_grad_enabled(False)
    else:
        project_model = DummyIdentity()

    class Streamer(nn_tilde.Module):

        def __init__(self) -> None:
            super().__init__()

            self.net = blender.net
            self.encoder = blender.encoder
            self.post_encoder = blender.post_encoder
            self.encoder_time = blender.encoder_time
            self.project_model = project_model
            self.chunk_size = model_chunk_size
            self.zs_channels = zs_channels
            self.zt_channels = zt_channels
            self.ae_latents = ae_latents
            self.emb_model_structure = torch.jit.load(emb_model_path).eval()
            self.audio_channels = getattr(
                getattr(self.emb_model_structure, "model", None),
                "audio_channels", 1)
            audio_input_labels = [
                f"(signal) Input audio structure {i + 1}"
                for i in range(self.audio_channels)
            ]
            audio_output_labels = [
                f"(signal) Audio output {i + 1}"
                for i in range(self.audio_channels)
            ]
            self.drop_value = blender.drop_value
            self.latent_range = FLAGS.latent_range

            # Get the ae ratio
            dummy = torch.zeros(1, self.audio_channels, 4 * 4096)
            z = self.emb_model_structure.encode(dummy)
            self.ae_ratio = 4 * 4096 // z.shape[-1]

            self.sr = gin.query_parameter("%SR")

            ## ATTRIBUTES ##
            self.register_attribute("nb_steps", 2)
            self.register_attribute("guidance_structure", 1.)

            ## BUFFERS ##
            self.register_buffer("_device_tracker", torch.zeros(1))
            self.register_buffer(
                "t_values_cache",
                torch.linspace(0, 1, self.nb_steps[0] + 1, device=self.device))

            # Removed timbre for the moment as it requires two encoders
            # self.zt_buffer = self.n_signal_timbre * self.ae_ratio
            # self.emb_model_timbre = torch.jit.load(FLAGS.emb_model_path).eval()
            # ## BUFFERS ##
            # self.register_buffer(
            #     "previous_timbre",
            #     torch.zeros(4, self.ae_latents, self.n_signal_timbre))

            ## METHODS ##
            self.register_method(
                "structure",
                in_channels=self.audio_channels,
                in_ratio=1,
                out_channels=self.zs_channels,
                out_ratio=self.ae_ratio,
                input_labels=audio_input_labels,
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
                "diffuse_timbre",
                in_channels=self.audio_channels + zt_channels,
                in_ratio=1,
                out_channels=self.ae_latents,
                out_ratio=self.ae_ratio,
                input_labels=audio_input_labels + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ],
                output_labels=[
                    f"(signal) Latent output {i}"
                    for i in range(self.ae_latents)
                ],
                test_buffer_size=self.ae_ratio,
            )

            self.register_method(
                "diffuse_timbre_modulate",
                in_channels=self.audio_channels + zt_channels + zs_channels * 2,
                in_ratio=1,
                out_channels=self.ae_latents,
                out_ratio=self.ae_ratio,
                input_labels=audio_input_labels + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ] + [
                    f"(signal_{i}) Modulate structure"
                    for i in range(2 * self.zs_channels)
                ],
                output_labels=[
                    f"(signal) Latent output {i}"
                    for i in range(self.ae_latents)
                ],
                test_buffer_size=self.ae_ratio,
            )

            self.register_method(
                "generate",
                in_channels=zt_channels + zs_channels,
                in_ratio=self.ae_ratio,
                out_channels=self.audio_channels,
                out_ratio=1,
                input_labels=[
                    f"(signal) Input structure {i}" for i in range(zs_channels)
                ] + [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=audio_output_labels,
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "generate_timbre",
                in_channels=zt_channels + self.audio_channels,
                in_ratio=1,
                out_channels=self.audio_channels,
                out_ratio=1,
                input_labels=audio_input_labels + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ],
                output_labels=audio_output_labels,
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "generate_timbre_modulate",
                in_channels=zt_channels + self.audio_channels + 2 * zs_channels,
                in_ratio=1,
                out_channels=self.audio_channels,
                out_ratio=1,
                input_labels=audio_input_labels + [
                    f"(signal_{i}) Input timbre"
                    for i in range(self.zt_channels)
                ] + [
                    f"(signal_{i}) Modulate structure"
                    for i in range(2 * self.zs_channels)
                ],
                output_labels=audio_output_labels,
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "decode",
                in_channels=self.ae_latents,
                in_ratio=self.ae_ratio,
                out_channels=self.audio_channels,
                out_ratio=1,
                input_labels=[
                    f"(signal) Latent input {i}"
                    for i in range(self.ae_latents)
                ],
                output_labels=audio_output_labels,
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
                test_buffer_size=128,
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
                test_buffer_size=128,
            )

        @property
        def device(self):
            return self._device_tracker.device

        @torch.jit.export
        def get_guidance_timbre(self) -> float:
            print("Depreciated")
            return 0.

        @torch.jit.export
        def set_guidance_timbre(self, guidance_timbre: float) -> int:
            print("Depreciated")
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
            self.t_values_cache = torch.linspace(0,
                                                 1,
                                                 nb_steps + 1,
                                                 device=self.device)
            return 0

        def model_forward(self, x: torch.Tensor, time: torch.Tensor,
                          cond: torch.Tensor, time_cond: torch.Tensor,
                          cache_index: int) -> torch.Tensor:

            if self.guidance_structure[0] == 1.:
                dx = self.net(x,
                              time=time,
                              cond=cond,
                              time_cond=time_cond,
                              cache_index=cache_index)
                return dx
            else:
                dx = self.net(x.repeat(2, 1, 1),
                              time=time.repeat(2, 1, 1),
                              cond=cond.repeat(2, 1),
                              time_cond=torch.cat([
                                  time_cond,
                                  self.drop_value * torch.ones_like(time_cond),
                              ]),
                              cache_index=cache_index)

                dx_full, dx_none = torch.chunk(dx, 2, dim=0)

                dx = dx_none + self.guidance_structure[0] * (dx_full - dx_none)

                return dx

        def sample(self, x_last: torch.Tensor, cond: torch.Tensor,
                   time_cond: torch.Tensor):

            dt = 1 / self.nb_steps[0]

            for i, t in enumerate(self.t_values_cache[:-1]):
                t = t.repeat(x_last.shape[0])

                x_last = x_last + dt * self.model_forward(x_last,
                                                time=t,
                                                cond=cond,
                                                cache_index=i,
                                                time_cond=time_cond)

                self.net.roll_cache(x_last.shape[-1], i)
            return x_last

        # @torch.jit.export
        # def timbre(self, x) -> torch.Tensor:
        #     x = self.emb_model_timbre.encode(x)

        #     self.previous_timbre[:x.shape[0]] = torch.cat(
        #         (self.previous_timbre[:x.shape[0]], x), -1)[..., x.shape[-1]:]

        #     zsem = self.encoder.forward_stream(
        #         self.previous_timbre[:x.shape[0]])

        #     if self.post_encoder is not None:
        #         zsem = self.post_encoder(zsem)

        #     zsem = zsem / self.latent_range

        #     return zsem.unsqueeze(-1).repeat((1, 1, self.chunk_size))

        @torch.jit.export
        def structure(self, x) -> torch.Tensor:
            n = x.shape[0]
            x = self.emb_model_structure.encode(x[:1])
            x = self.encoder_time.forward_stream(x).repeat(n, 1, 1)

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

        def modulate(selx, time_cond: torch.Tensor, modulators: torch.Tensor):
            """
            Apply FiLM-style modulation where modulators are organized as:
            [scale1, bias1, scale2, bias2, ...]
            """
            # Split interleaved scale/bias pairs
            scales = modulators[:, 0::2, :]  # take even indices → scale_i
            shifts = modulators[:, 1::2, :]  # take odd indices  → bias_i
            # Apply modulation
            time_cond = time_cond * scales + shifts
            return time_cond

        @torch.jit.export
        def diffuse_timbre_modulate(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            timbre_start = self.audio_channels
            timbre_end = timbre_start + self.zt_channels
            zsem = x[:, timbre_start:timbre_end].mean(-1)
            zsem = zsem * self.latent_range

            audio = x[:, :self.audio_channels, :]
            time_cond = self.structure(audio)

            modulators = x[:, timbre_end:, :]

            modulators = torch.nn.functional.interpolate(modulators,
                                                         scale_factor=1 /
                                                         self.ae_ratio,
                                                         mode="nearest")

            time_cond = self.modulate(time_cond, modulators)

            x = torch.randn(n, self.ae_latents, time_cond.shape[-1])
            x = self.sample(x, time_cond=time_cond, cond=zsem)

            return x

        @torch.jit.export
        def diffuse_timbre(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            zsem = x[:, self.audio_channels:].mean(-1)
            zsem = zsem * self.latent_range

            audio = x[:, :self.audio_channels, :]
            time_cond = self.structure(audio)

            x = torch.randn(n, self.ae_latents, time_cond.shape[-1])
            x = self.sample(x, time_cond=time_cond, cond=zsem)

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

        @torch.jit.export
        def generate_timbre_modulate(self, x: torch.Tensor) -> torch.Tensor:
            z = self.diffuse_timbre_modulate(x)
            audio = self.decode(z)
            return audio

        @torch.jit.export
        def map2latent(self, x: torch.Tensor) -> torch.Tensor:
            tdim = x.shape[-1]
            mapvec = x.mean(-1)
            latents = self.project_model.decode(mapvec)
            return latents.unsqueeze(-1).repeat((1, 1, tdim))

        @torch.jit.export
        def latent2map(self, x: torch.Tensor) -> torch.Tensor:
            tdim = x.shape[-1]
            latents = x.mean(-1)
            map = self.project_model.encode(latents)
            return map.unsqueeze(-1).repeat((1, 1, tdim))

    ####

    streamer = Streamer()

    # Some tests
    dummmy = torch.randn(
        1, streamer.audio_channels + 2 * zs_channels + zt_channels, 8192)
    out = streamer.diffuse_timbre_modulate(dummmy)

    dummmy = torch.randn(1, zs_channels + zt_channels, model_chunk_size)
    out = streamer.diffuse(dummmy)
    out_name = os.path.join(folder,
                            "after.audio." + folder.split("/")[-1] + ".ts")

    streamer.export_to_ts(out_name)

    out_name_plot = os.path.join(
        folder, "after.audio." + folder.split("/")[-1] + ".png")

    out_name_plot_legend = os.path.join(
        folder, "after.audio." + folder.split("/")[-1] + "_legend.png")

    if FLAGS.latent_project:
        fig.savefig(out_name_plot,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor=fig.get_facecolor(),
                    transparent=False)

        legend_fig.savefig(out_name_plot_legend,
                           dpi=300,
                           bbox_inches='tight',
                           pad_inches=0.1,
                           facecolor=fig.get_facecolor(),
                           transparent=False)

    print("Bravo - Export successful")


if __name__ == "__main__":
    app.run(main)
