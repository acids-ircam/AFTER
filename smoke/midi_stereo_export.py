"""Exercise MIDI mono/stereo export on CPU, without training or model downloads.

Run from the repo root with its dependencies installed:
    python smoke/midi_stereo_export.py

Creates tiny random RectifiedFlow checkpoints and scripted codec fixtures, then
runs the real MIDI exporter in fresh processes. This tests export/streaming I/O,
not trained audio quality. Artifacts are retained in the printed directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gin
import nn_tilde
from after.diffusion import RectifiedFlow

RATIO = 4096
LATENTS = 8
VOICES = 8
TIMBRES = 2


class CodecCore(nn.Module):
    def __init__(self, channels, decoder_channels):
        super().__init__()
        self.audio_channels = channels
        self.decoder_channels = decoder_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.audio_channels, "codec input channels"
        z = x.mean(1, keepdim=True).reshape(x.shape[0], 1, -1, 4096).mean(-1)
        return z.repeat(1, 8, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :self.decoder_channels].repeat_interleave(4096, dim=-1)


class Codec(nn_tilde.Module):
    def __init__(self, channels, decoder_channels):
        super().__init__()
        self.model = CodecCore(channels, decoder_channels)
        self.register_method("encode", channels, 1, LATENTS, RATIO)
        self.register_method("decode", LATENTS, RATIO, decoder_channels, 1)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


class LegacyMonoCodec(nn_tilde.Module):
    """The legacy codec path has no model.audio_channels attribute."""
    def __init__(self):
        super().__init__()
        self.register_method("encode", 1, 1, LATENTS, RATIO)
        self.register_method("decode", LATENTS, RATIO, 1, 1)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 1, "legacy mono input channels"
        return x.reshape(x.shape[0], 1, -1, 4096).mean(-1).repeat(1, 8, 1)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :1].repeat_interleave(4096, dim=-1)


def export(source, folder, codec, env, log):
    setup = ("import runpy,sys,torch; torch.set_num_threads(1); "
             "torch.manual_seed(1234); source=sys.argv.pop(1); "
             "runpy.run_path(source,run_name='__main__')")
    command = [sys.executable, "-c", setup, str(source),
               "--model_path", str(folder), "--emb_model_path", str(codec),
               "--nolatent_project"]
    with log.open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT,
                                timeout=120)
    return result.returncode


def check_model(path, channels):
    model = torch.jit.load(str(path)).eval()
    assert model.audio_channels == channels
    assert model.sr == 44100
    assert model.generate_params.tolist() == [18, 1, channels, 1]
    assert model.diffuse_params.tolist() == [18, 1, LATENTS, RATIO]
    assert model.decode_params.tolist() == [LATENTS, RATIO, channels, 1]
    assert len(model.generate_output_labels) == channels
    assert len(model.decode_output_labels) == channels
    notes = torch.zeros(1, 2 * VOICES + TIMBRES, 4 * RATIO)
    notes[:, 0] = 60
    notes[:, 1, :2 * RATIO] = 100
    notes[:, 2] = 67
    notes[:, 3, RATIO:] = 80
    for steps, guidance in ((1, 1.0), (2, 2.0)):
        model.set_nb_steps(steps)
        model.set_guidance_structure(guidance)
        for _ in range(2):
            latents = model.diffuse(notes)
            assert latents.shape == (1, LATENTS, 4)
            audio = model.decode(latents)
            assert audio.shape == (1, channels, notes.shape[-1])
            assert torch.isfinite(latents).all() and torch.isfinite(audio).all()
            generated = model.generate(notes)
            assert generated.shape == audio.shape and torch.isfinite(generated).all()
            if channels == 2:
                assert not torch.equal(audio[:, 0], audio[:, 1]), "stereo collapsed"
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-exporter", type=Path,
                        help="Optional old export_midi.py for seeded mono regression")
    args = parser.parse_args()
    work = args.output_dir or Path(tempfile.mkdtemp(prefix="after-midi-export-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {work}", flush=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", ACCELERATE_USE_CPU="true",
               MPLCONFIGDIR=str(work / "matplotlib"),
               NUMBA_CACHE_DIR=str(work / "numba"))
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(1234)
    config = (ROOT / "smoke/midi_stereo_export.gin").read_text()
    gin.parse_config(config)
    checkpoint = RectifiedFlow().eval().state_dict()
    cases = []
    for name, channels, fixture in (
            ("mono", 1, Codec(1, 1)),
            ("stereo", 2, Codec(2, 2)),
            ("legacy_mono", 1, LegacyMonoCodec()),
            ("mismatched_decoder", 2, Codec(2, 1))):
        folder = work / name
        folder.mkdir()
        (folder / "config.gin").write_text(config)
        torch.save({"model_state": checkpoint}, folder / "checkpoint0_EMA.pt")
        codec = folder / "codec.ts"
        fixture.export_to_ts(str(codec))
        log = folder / "export.log"
        code = export(ROOT / "after_scripts/export_midi.py", folder, codec, env, log)
        case = {"name": name, "channels": channels,
                "returncode": code, "log": str(log)}
        if name == "mismatched_decoder":
            assert code != 0, "mismatched stereo decoder was accepted"
            assert "Wrong number of output channels" in log.read_text()
        else:
            assert code == 0, log.read_text()
            artifact = folder / f"after.midi.{name}.ts"
            check_model(artifact, channels)
            if name == "mono" and args.baseline_exporter:
                patched = folder / "patched.ts"
                artifact.rename(patched)
                assert export(args.baseline_exporter, folder, codec, env,
                              folder / "baseline.log") == 0
                old = torch.jit.load(str(artifact)).eval()
                new = torch.jit.load(str(patched)).eval()
                x = torch.zeros(1, 18, 4 * RATIO)
                x[:, 0], x[:, 1] = 60, 100
                for call in range(3):
                    torch.manual_seed(4321 + call)
                    expected = old.generate(x)
                    torch.manual_seed(4321 + call)
                    actual = new.generate(x)
                    assert torch.equal(actual, expected), "mono output changed"
                artifact.unlink()
                patched.rename(artifact)
                case["seeded_mono_backward_equivalence"] = True
            if name == "stereo" and args.baseline_exporter:
                old_folder = work / "baseline_stereo"
                old_folder.mkdir()
                (old_folder / "config.gin").write_text(config)
                torch.save({"model_state": checkpoint}, old_folder / "checkpoint0_EMA.pt")
                old_log = old_folder / "export.log"
                assert export(args.baseline_exporter, old_folder, codec, env, old_log) != 0
                assert "codec input channels" in old_log.read_text()
                case["baseline_stereo_probe_fails"] = True
        cases.append(case)
    report = {"passed": True, "scope": "Export/streaming I/O with tiny random checkpoints and scripted codec fixtures; no training or audio-quality claim.",
              "codec_ratio": RATIO, "codec_latents": LATENTS,
              "voices": VOICES, "timbres": TIMBRES, "cases": cases}
    (work / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("MIDI mono/stereo export passed.", flush=True)


if __name__ == "__main__":
    main()
