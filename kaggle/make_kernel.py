#!/usr/bin/env python
"""Write a Kaggle kernel folder that runs one generation (+ evaluation) job.

    python kaggle/make_kernel.py --name evidgeo-sd35m-s0 --model sd35-medium \
        --gen-args "--n-prompts 300 --seeds 4 --traj-prompts 30 --shard 0/1"
    kaggle kernels push -p kaggle/build/evidgeo-sd35m-s0
    kaggle kernels status dpaul93/evidgeo-sd35m-s0
    kaggle kernels output dpaul93/evidgeo-sd35m-s0 -p ./kaggle_out/evidgeo-sd35m-s0

If HF_TOKEN is set in the local environment when this script runs, it is
written into the (private, gitignored) kernel build so gated models download;
otherwise the kernel looks for a Kaggle secret named HF_TOKEN.

The kernel clones the public repo, fetches the master caption manifest from
the repo's data/ folder, generates into /kaggle/working/gen, then runs
run_eval.py on those images so the (small) feature files come back with the
kernel output even if the images are too large to keep.
"""
import argparse
import json
import os
import textwrap

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="kernel slug, e.g. evidgeo-sd35m-s0")
ap.add_argument("--user", default="dpaul93")
ap.add_argument("--model", required=True)
ap.add_argument("--gen-args", default="")
ap.add_argument("--eval-args", default="--grids 7 --faith-per-bucket 0")
ap.add_argument("--keep-images", action="store_true", help="zip images into the output")
ap.add_argument("--branch", default="main")
ap.add_argument("--accelerator", default="", help="Kaggle machine_shape; empty = default GPU")
a = ap.parse_args()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", a.name)
os.makedirs(out, exist_ok=True)
code = f'''
import os, subprocess, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def sh(c):
    # tee everything into run.log so timings survive in the kernel output
    print("+", c, flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-c", f"({{c}}) 2>&1 | tee -a /kaggle/working/run.log"], check=True)
sh("git clone --depth 1 -b {a.branch} https://github.com/dpaul0501/evidgeocode.git /kaggle/working/repo")
os.chdir("/kaggle/working/repo")
# Kaggle's preinstalled torchao is older than diffusers expects and breaks its imports; not needed here
sh("pip -q uninstall -y torchao || true")
sh("pip -q install -U 'diffusers>=0.37.1' transformers accelerate sentencepiece protobuf pyarrow statsmodels datasets")
sh("python -c 'import diffusers, transformers, torch; print(\\"versions:\\", diffusers.__version__, transformers.__version__, torch.__version__)'")
{"os.environ.setdefault('HF_TOKEN', " + repr(os.environ['HF_TOKEN']) + ")" if os.environ.get("HF_TOKEN") else "# HF_TOKEN not injected at build time"}
if os.environ.get("HF_TOKEN") is None:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("no HF_TOKEN secret (needed for gated models):", e)
M = "data/manifest_master_N5000_seed42.jsonl"
G = "/kaggle/working/gen_{a.name}"
sh(f"python scripts/generate_modern.py --model {a.model} --master-manifest {{M}} --out {{G}} {a.gen_args}")
sh(f"python scripts/run_eval.py --guidance-roots {{G}} --trajectory-roots {{G}} --master-manifest {{M}} "
   f"--real --out /kaggle/working/eval --device cuda {a.eval_args}")
sh("python scripts/analyze.py --run-dir /kaggle/working/eval")
{"sh(f'cd /kaggle/working && zip -qr images.zip {{os.path.basename(G)}}')" if a.keep_images else ""}
sh(f"rm -rf {{G}}/images {{G}}/traj /kaggle/working/eval/real_cache /kaggle/working/repo")
'''
with open(os.path.join(out, "kernel.py"), "w") as f:
    f.write(textwrap.dedent(code))
meta = {
    "id": f"{a.user}/{a.name}", "title": a.name, "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "machine_shape": a.accelerator,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
with open(os.path.join(out, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", out)
