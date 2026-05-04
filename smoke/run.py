"""Smoke runner for the AFTER training pipelines.

Two modes:

* `--mode ae` (default) — chunks audio into a waveform-only LMDB and runs
  `after train_autoencoder` for ~200 steps. No pretrained weights required.
* `--mode diffusion` — uses a pretrained autoencoder TorchScript (the
  offline `.ts` file) to precompute latent codes, then runs `after train`
  for one short epoch.

Run from the repo root with the venv activated (or invoking the venv's
`after` console script directly):

    .venv/bin/python smoke/run.py --mode ae \\
        --input_path "/path/to/audio/dir"

    .venv/bin/python smoke/run.py --mode diffusion \\
        --input_path "/path/to/audio/dir" \\
        --emb_model_path "/path/to/codec.ts"

Output goes under `--output_dir/<mode>/`. Each stage runs in a fresh
subprocess so absl flag definitions don't collide between prepare_dataset
and the train scripts.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = Path(__file__).resolve().parent
AE_GIN = SMOKE_DIR / "smoke.gin"
DIFFUSION_GIN = SMOKE_DIR / "diffusion_smoke.gin"
AFTER_BIN = REPO_ROOT / ".venv" / "bin" / "after"


def _run(stage: str, args: list[str], device: str) -> None:
    env = os.environ.copy()
    # accelerate picks MPS on Apple Silicon and trips a torch assert
    # ("invalid low watermark ratio 1.4") when torch's high watermark
    # ratio is below accelerate's default low (1.4). 0.0 disables the
    # upper cap. Direct assignment (not setdefault) overwrites a
    # user-set value.
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if device == "cpu":
        env["ACCELERATE_USE_CPU"] = "true"
    elif device == "mps":
        env.pop("ACCELERATE_USE_CPU", None)
    cmd = [str(AFTER_BIN), stage, *args]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    print(f"[smoke] stage {stage!r} (device={device}) took "
          f"{time.perf_counter() - t0:.1f}s", flush=True)


def _prepare_args_ae(input_path: str, dataset_dir: str) -> list[str]:
    return [
        "--input_path", input_path,
        "--output_path", dataset_dir,
        "--save_waveform", "True",
        "--waveform_augmentation", "none",
        "--device", "cpu",
    ]


def _train_args_ae(dataset_dir: str, runs_dir: str, device: str) -> list[str]:
    return [
        "--name", "smoke_ae",
        "--save_dir", runs_dir,
        "--db_path", dataset_dir,
        "--config", str(AE_GIN),
        "--bsize", "2",
        "--num_signal", "65536",
        "--device", device,
        "--num_workers", "0",
    ]


def _prepare_args_diffusion(input_path: str, dataset_dir: str,
                            emb_model_path: str) -> list[str]:
    return [
        "--input_path", input_path,
        "--output_path", dataset_dir,
        "--emb_model_path", emb_model_path,
        "--waveform_augmentation", "none",
        "--device", "cpu",
    ]


def _train_args_diffusion(dataset_dir: str, runs_dir: str,
                          emb_model_path: str, device: str) -> list[str]:
    return [
        "--name", "smoke_diffusion",
        "--out_path", runs_dir,
        "--db_path", dataset_dir,
        "--emb_model_path", emb_model_path,
        "--config", str(DIFFUSION_GIN),
        "--bsize", "2",
        "--n_signal", "32",
        "--device", device,
        "--num_workers", "0",
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("ae", "diffusion"), default="ae",
                   help="Which pipeline to smoke-test.")
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto",
                   help="Compute device for the train stage.")
    p.add_argument("--input_path", required=True,
                   help="Directory containing audio files for the smoke run.")
    p.add_argument("--emb_model_path",
                   help="Path to the pretrained autoencoder TorchScript "
                        "(required for --mode diffusion). Use the offline "
                        "*.ts file, not the *_stream.ts file.")
    p.add_argument("--output_dir", default=str(REPO_ROOT / "smoke" / "output"),
                   help="Where to put the prepared LMDB + training runs.")
    args = p.parse_args()

    if args.mode == "diffusion" and not args.emb_model_path:
        p.error("--emb_model_path is required for --mode diffusion")
    if not AFTER_BIN.exists():
        p.error(f"Expected {AFTER_BIN} (run `pip install -e .` in the venv)")

    output_dir = Path(args.output_dir).resolve() / args.mode
    dataset_dir = output_dir / "dataset"
    runs_dir = output_dir / "runs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "ae":
        prepare_args = _prepare_args_ae(args.input_path, str(dataset_dir))
        train_args = _train_args_ae(str(dataset_dir), str(runs_dir), args.device)
        train_stage = "train_autoencoder"
    else:
        prepare_args = _prepare_args_diffusion(args.input_path, str(dataset_dir),
                                               args.emb_model_path)
        train_args = _train_args_diffusion(str(dataset_dir), str(runs_dir),
                                           args.emb_model_path, args.device)
        train_stage = "train"

    if not (dataset_dir / "data.mdb").exists():
        _run("prepare_dataset", prepare_args, device="cpu")
    else:
        print(f"[smoke] dataset already at {dataset_dir}, skipping prepare")

    _run(train_stage, train_args, device=args.device)


if __name__ == "__main__":
    main()
