import nn_tilde
import torch
import cached_conv as cc
from after.autoencoder import AutoEncoder

import gin
from absl import app, flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer("step", None, "Step to load the model from")
flags.DEFINE_string("model_path", None, "Path of the trained model")


class AE_notcausal(nn_tilde.Module):

    def __init__(self) -> None:
        super().__init__()

        sr = gin.query_parameter("%SR")

        config = os.path.join(FLAGS.model_path, "config.gin")

        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                [config],
                [],
            )

        model = AutoEncoder()

        if FLAGS.step is None:
            files = os.listdir(FLAGS.model_path)
            files = [f for f in files if f.startswith("checkpoint")]
            steps = [f.replace("checkpoint", "")[:-3] for f in files]
            step = max([int(s) for s in steps])
            checkpoint_file = "checkpoint" + str(step) + ".pt"
        else:
            checkpoint_file = "checkpoint" + str(FLAGS.step) + ".pt"

        print("Export using : ", checkpoint_file)

        d = torch.load(os.path.join(FLAGS.model_path, checkpoint_file))

        model.load_state_dict(d["model_state"], strict=False)

        self.model = model

        test_array = torch.zeros((3, 1, 131072))
        z, _ = self.model.encode(test_array)
        y = self.model.decode(z)

        assert test_array.shape[-1] == y.shape[-1]

        self.comp_ratio = test_array.shape[-1] // z.shape[-1]
        self.n_fade = 4

        self.latent_size = gin.query_parameter("%LATENT_SIZE")
        self.target_channels = 1

        self.register_buffer("out_buffer",
                             torch.zeros(4, 1, self.comp_ratio * self.n_fade))
        self.register_buffer("z_buffer",
                             torch.zeros(4, self.latent_size, self.n_fade))

        self.register_method(
            "encode",
            in_channels=self.target_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=self.comp_ratio,
            input_labels=['(signal) input 1'],
            output_labels=[f"latent {i}" for i in range(self.latent_size)],
            test_buffer_size=self.comp_ratio,
        )

        self.register_method("decode",
                             in_channels=self.latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=self.target_channels,
                             out_ratio=1,
                             input_labels=[
                                 f'(signal) Latent dimension {i+1}'
                                 for i in range(self.latent_size)
                             ],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

        self.register_method("forward",
                             in_channels=1,
                             in_ratio=1,
                             out_channels=1,
                             out_ratio=1,
                             input_labels=['(signal) input 1'],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)

        z = self.model.encoder(x)

        z, _ = self.model.bottleneck(z)

        x = self.model.decoder(z)

        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)

        return x

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)
        z = self.model.encoder(x)
        z, _ = self.model.bottleneck(z)

        return z

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        n = z.shape[0]

        z = torch.cat((self.z_buffer[:n], z), -1)

        x = self.model.decoder(z)
        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)

        self.z_buffer[:n] = z[:n, :, -self.n_fade:].clone()

        alpha = torch.linspace(0, 1,
                               self.n_fade * self.comp_ratio)[None,
                                                              None, :].to(x)

        x[..., :self.comp_ratio *
          self.n_fade] = (1 - alpha) * self.out_buffer[:n].to(x) + alpha * x[
              ..., :self.comp_ratio * self.n_fade]

        self.out_buffer[:n] = x[:n, :, -self.comp_ratio * self.n_fade:].clone()

        x = x[..., :-self.comp_ratio * self.n_fade]

        return x


class AE_causal(nn_tilde.Module):

    def __init__(self) -> None:
        super().__init__()

        sr = gin.query_parameter("%SR")

        config = os.path.join(FLAGS.model_path, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                [config],
                [],
            )

        model = AutoEncoder()

        if FLAGS.step is None:
            files = os.listdir(FLAGS.model_path)
            files = [f for f in files if f.startswith("checkpoint")]
            steps = [f.replace("checkpoint", "")[:-3] for f in files]
            step = max([int(s) for s in steps])
            checkpoint_file = "checkpoint" + str(step) + ".pt"
        else:
            checkpoint_file = "checkpoint" + str(FLAGS.step) + ".pt"

        print("Export using : ", checkpoint_file)

        d = torch.load(os.path.join(FLAGS.model_path, checkpoint_file))

        model.load_state_dict(d["model_state"], strict=False)

        self.model = model

        test_array = torch.zeros((3, 1, 131072))
        z, _ = self.model.encode(test_array)
        y = self.model.decode(z)
        assert test_array.shape[-1] == y.shape[-1]

        self.comp_ratio = test_array.shape[-1] // z.shape[-1]

        self.latent_size = gin.query_parameter("%LATENT_SIZE")
        self.target_channels = 1

        self.register_method(
            "encode",
            in_channels=self.target_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=self.comp_ratio,
            input_labels=['(signal) input 1'],
            output_labels=[f"latent {i}" for i in range(self.latent_size)],
            test_buffer_size=self.comp_ratio,
        )

        self.register_method("decode",
                             in_channels=self.latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=self.target_channels,
                             out_ratio=1,
                             input_labels=[
                                 f'(signal) Latent dimension {i+1}'
                                 for i in range(self.latent_size)
                             ],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

        self.register_method("forward",
                             in_channels=1,
                             in_ratio=1,
                             out_channels=1,
                             out_ratio=1,
                             input_labels=['(signal) input 1'],
                             output_labels=[
                                 '(signal) Channel %d' % d
                                 for d in range(1, self.target_channels + 1)
                             ])

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)

        z = self.model.encoder(x)

        z, _ = self.model.bottleneck(z)

        x = self.model.decoder(z)

        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)

        return x

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.use_pqmf:
            x = self.model.pqmf(x)
        z = self.model.encoder(x)
        z, _ = self.model.bottleneck(z)

        return z

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.model.decoder(z)
        if self.model.use_pqmf:
            x = self.model.pqmf.inverse(x)
        return x


def main(argv):

    config = os.path.join(FLAGS.model_path, "config.gin")

    with gin.unlock_config():
        gin.parse_config_files_and_bindings(
            [config],
            [],
        )
    try:
        is_causal = gin.query_parameter("convs.get_padding.mode") == "causal"
    except Exception as e:
        print("Error in gin query: ", e)
        print("This is normal if your model is not causal")
        is_causal = False

    print("is causal: ", is_causal)

    cc.use_cached_conv(False)
    ae = AE_causal()
    test_array = torch.zeros((3, 1, 131072))
    z = ae.encode(test_array)
    x = ae.decode(z)
    ae.export_to_ts(os.path.join(FLAGS.model_path, "export.ts"))

    if is_causal:
        gin.bind_parameter("SimpleNetsStream.CachedGroupNorm.stream", True)
        cc.use_cached_conv(True)
        ae_stream = AE_causal()

        test_array = torch.zeros((3, 1, 131072))
        z = ae_stream.encode(test_array)
        x = ae_stream.decode(z)

        ae_stream.export_to_ts(
            os.path.join(FLAGS.model_path, "export_stream.ts"))
        print("Success !")

    else:
        gin.bind_parameter("SimpleNetsStream.CachedGroupNorm.stream", True)
        cc.use_cached_conv(True)
        ae_encode = AE_notcausal()
        cc.use_cached_conv(False)
        ae_stream = AE_notcausal()
        ae_stream.model.encoder = ae_encode.model.encoder

        test_array = torch.zeros((3, 1, 131072))
        z = ae_stream.encode(test_array)
        x = ae_stream.decode(z)

        ae_stream.export_to_ts(
            os.path.join(FLAGS.model_path, "export_stream.ts"))
        print("Success !")


if __name__ == "__main__":
    app.run(main)
