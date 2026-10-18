"""Synthetic stereo pipeline smoke test; run with `conda run -n after python smoke/stereo_pipeline.py`.

Runs preprocessing, three codec updates (including a discriminator update), codec exports,
latent preprocessing, four AFTER updates, and AFTER export on CPU.
Artifacts are kept in the printed temporary directory for inspection.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def run(stage, *args):
    print(f"\n=== {stage} ===", flush=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", ACCELERATE_USE_CPU="true")
    subprocess.run(
        [sys.executable, "-m", f"after_scripts.{stage}", *map(str, args)],
        cwd=ROOT, env=env, check=True,
    )


def check_audio(x, channels, length):
    assert x.shape == (1, channels, length), x.shape
    assert torch.isfinite(x).all()


def main():
    torch.set_num_threads(1)
    work = Path(tempfile.mkdtemp(prefix="after-stereo-"))
    print(f"Artifacts: {work}", flush=True)
    os.environ["NUMBA_CACHE_DIR"] = str(work / "numba_cache")
    os.environ["MPLCONFIGDIR"] = str(work / "matplotlib")
    audio = work / "audio"
    audio.mkdir()
    t = np.arange(65536) / 44100
    for i in range(6):
        left = 0.2 * np.sin(2 * np.pi * (220 + 30 * i) * t)
        right = 0.1 * np.sin(2 * np.pi * (330 + 30 * i) * t)
        waveform = left if i % 2 == 0 else np.stack((left, right), axis=-1)
        sf.write(audio / f"{'mono' if i % 2 == 0 else 'stereo'}_{i}.wav", waveform, 44100)

    waveform_db = work / "waveforms"
    run("prepare_dataset", "--input_path", audio, "--output_path", waveform_db,
        "--stereo", "--save_waveform", "--num_signal", 65536,
        "--num_augments", 0, "--device", "cpu", "--db_size", 1)

    from after.dataset import SimpleDataset
    dataset = SimpleDataset(path=str(waveform_db), keys=["waveform", "metadata"])
    assert len(dataset) == 6
    for item in dataset:
        x = item["waveform"]
        assert x.shape == (2, 65536), x.shape
        if "mono" in item["metadata"]["path"]:
            np.testing.assert_array_equal(x[0], x[1])
        else:
            assert not np.array_equal(x[0], x[1])

    codec_dir = work / "codec_runs" / "tiny"
    run("train_autoencoder", "--name", "tiny", "--out_path", codec_dir.parent,
        "--db_path", waveform_db, "--config", ROOT / "smoke/stereo_ae.gin",
        "--stereo", "--batch_size", 2, "--n_signal", 32768,
        "--device", "cpu", "--num_workers", 0, "--nouse_validation", "--nouse_psts")
    first_checkpoint = torch.load(codec_dir / "checkpoint0.pt", map_location="cpu")
    last_checkpoint = torch.load(codec_dir / "checkpoint2.pt", map_location="cpu")
    first = first_checkpoint["model_state"]
    last = last_checkpoint["model_state"]
    for head in ("stereo_merge", "stereo_split"):
        keys = [k for k in first if k.startswith(head + ".")]
        assert keys and any(not torch.equal(first[k], last[k]) for k in keys), head
    assert any(
        not torch.equal(value, last_checkpoint["dis_state"][key])
        for key, value in first_checkpoint["dis_state"].items()
    ), "Discriminator parameters did not update"
    run("export_autoencoder", "--model_path", codec_dir, "--step", 2)
    for filename in ("export.ts", "export_stream.ts"):
        codec = torch.jit.load(str(codec_dir / filename)).eval()
        with torch.no_grad():
            assert codec.encode_params.tolist() == [2, 1, 8, 4096]
            assert codec.decode_params.tolist() == [8, 4096, 2, 1]
            z = codec.encode(torch.randn(1, 2, 32768))
            assert z.shape == (1, 8, 8), z.shape
            check_audio(codec.decode(z), 2, 32768)

    latent_db = work / "latents"
    run("prepare_dataset", "--input_path", audio, "--output_path", latent_db,
        "--stereo", "--emb_model_path", codec_dir / "export.ts",
        "--num_signal", 65536, "--num_augments", 0,
        "--device", "cpu", "--db_size", 1)
    latents = SimpleDataset(path=str(latent_db), keys=["z"])
    assert len(latents) == 6
    for item in latents:
        assert item["z"].shape == (8, 16), item["z"].shape
        assert np.isfinite(item["z"]).all()

    after_dir = work / "after_runs" / "tiny"
    run("train", "--name", "tiny", "--out_path", after_dir.parent,
        "--db_path", latent_db, "--emb_model_path", codec_dir / "export.ts",
        "--config", ROOT / "smoke/stereo_diffusion.gin", "--batch_size", 2,
        "--n_signal", 8, "--device", "cpu", "--num_workers", 0,
        "--nouse_validation")
    steps = [int(p.name.split("_")[0].replace("checkpoint", ""))
             for p in after_dir.glob("checkpoint*_EMA.pt")]
    assert 1 <= max(steps) <= 3, steps
    run("export", "--model_path", after_dir,
        "--emb_model_path", codec_dir / "export_stream.ts", "--nolatent_project")
    model = torch.jit.load(str(after_dir / "after.audio.tiny.ts")).eval()
    assert model.structure_params.tolist() == [2, 1, 2, 4096]
    assert model.decode_params.tolist() == [8, 4096, 2, 1]
    assert model.generate_params.tolist() == [4, 4096, 2, 1]
    assert model.diffuse_timbre_params.tolist() == [4, 1, 8, 4096]
    assert model.diffuse_timbre_modulate_params.tolist() == [8, 1, 8, 4096]
    assert model.generate_timbre_params.tolist() == [4, 1, 2, 1]
    assert model.generate_timbre_modulate_params.tolist() == [8, 1, 2, 1]
    with torch.no_grad():
        audio_input = torch.randn(1, 2, 32768)
        structure = model.structure(audio_input)
        assert structure.shape == (1, 2, 8), structure.shape
        latent_input = torch.cat((structure, torch.zeros_like(structure)), dim=1)
        z = model.diffuse(latent_input)
        check_audio(model.decode(z), 2, 32768)
        check_audio(model.generate(latent_input), 2, 32768)
        timbre_input = torch.cat((audio_input, torch.zeros_like(audio_input)), dim=1)
        assert model.diffuse_timbre(timbre_input).shape == (1, 8, 8)
        check_audio(model.generate_timbre(timbre_input), 2, 32768)
        modulators = torch.zeros(1, 4, 32768)
        modulators[:, 0::2] = 1
        modulated_input = torch.cat((timbre_input, modulators), dim=1)
        assert model.diffuse_timbre_modulate(modulated_input).shape == (1, 8, 8)
        check_audio(model.generate_timbre_modulate(modulated_input), 2, 32768)
    print(f"\nStereo pipeline passed. Artifacts: {work}")


if __name__ == "__main__":
    main()
