"""One bounded self-evolution round of AgenticArXiv-RL on a free Kaggle GPU (T4 x1).

Run inside a Kaggle notebook (Settings: GPU T4, Internet ON) with a cell:

    !curl -sL https://raw.githubusercontent.com/hazelxue/AgenticArXiv-RL/kaggle/round1/kaggle_round1.py -o /kaggle/working/kaggle_round1.py && python /kaggle/working/kaggle_round1.py

Expected Kaggle Dataset "agenticarxiv-round1" (upload from F:/oss-work/pr2_data + the snapshot):
    mock_arxiv_snapshot.json                     offline replay snapshot (build_snapshot output)
    sft_v3_train_mix.jsonl(.manifest.json)       base train mix (1020 original + 2628 parametric rows)
    sft_v2_parametric_seed.jsonl.manifest.json   derived -> parent lineage for `self_evolve run`
    kaggle_round1.py                             this file

Every step writes a marker so a restarted session resumes where it stopped (Kaggle caps a
session at 12 h). Measured on a T4: 14-20 s per optimizer step with fp32 master weights at
max_length 3072, so 2 epochs x 2 rounds (1890 steps, ~9 h) plus evals would overrun the cap;
1 epoch per round fits.

Resume after a failed version: Kaggle keeps /kaggle/working of a failed run as that version's
Output. Attach it to the notebook (Add Input -> Your Work -> this notebook's output) and rerun;
setup() restores the step markers, the eval/artifact directories and the saved models from it.

Why Qwen2.5-0.5B and full fine-tuning: the repo's frozen QLoRA recipe requires bf16, which the
T4/P100 do not support, and a 1.5B full fine-tune does not fit 16 GB. The round is still a
complete, honest exercise of the loop; the PR reports whatever the gate decides.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_input() -> Path:
    # Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/ (new layout) or
    # /kaggle/input/<slug>/ (old layout); accept either.
    candidates = [Path("/kaggle/input/agenticarxiv-round1")]
    candidates += list(Path("/kaggle/input").glob("datasets/*/agenticarxiv-round1"))
    for candidate in candidates:
        if (candidate / "kaggle_round1.py").exists():
            return candidate
    raise SystemExit("dataset 'agenticarxiv-round1' is not attached to this notebook (Add Input)")


INPUT = _find_input()
REQUIRED_INPUTS = (
    "mock_arxiv_snapshot.json",
    "sft_v3_train_mix.jsonl",
    "sft_v3_train_mix.jsonl.manifest.json",
    "sft_v2_parametric_seed.jsonl.manifest.json",
)
_missing = [name for name in REQUIRED_INPUTS if not (INPUT / name).exists()]
if _missing:
    # A dataset "New Version" that only re-uploads the script silently drops the data files.
    raise SystemExit(
        f"dataset at {INPUT} is missing {_missing}; upload a new dataset version containing "
        "all 5 files from F:/oss-work/kaggle_dataset and re-attach the latest version"
    )
WORK = Path("/kaggle/working")
REPO = WORK / "AgenticArXiv-RL"
MARKS = WORK / "marks"
# Trainer checkpoints (fp32 weights + Adam states, ~6 GB each, up to 4 on disk at once) do not fit
# the 19.5 GB /kaggle/working quota; /kaggle/temp is the larger scratch disk. Only the final
# models (2 GB each) and the small artifacts are copied back into /kaggle/working.
OUTPUTS = Path("/kaggle/temp/outputs")
SAVED_MODELS = WORK / "outputs"

# kaggle/round1 = PR-2 branch (self_evolve mix + targeted wrapper set) + the trainable-model
# dtype fix (merged upstream as #106): without it --no-qlora loads Qwen in bf16 and the T4
# (compute capability 7.5) fails on the first matmul.
REPO_BRANCH = "kaggle/round1"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SPLIT = "data/splits/v3_81.json"
SNAPSHOT = "data/mock_arxiv_snapshot.json"
PARAMETRIC_MANIFEST = "data/sft/sft_v2_parametric_seed.jsonl.manifest.json"
EVAL_SPLITS = ("dev", "iid_test", "ood_test")
# max_length 3072 > longest sample (2676 tokens), so nothing is truncated, while the fp32 logits
# of a 152k vocab stay inside the T4's 16 GB.
TRAIN_ARGS = [
    "--no-qlora", "--epochs", "1", "--batch_size", "1", "--grad_accum", "8", "--lr", "1e-5",
    "--max_length", "3072", "--optim", "adamw_torch", "--no-verify", "--report_to", "none",
]

# Replays every held-out task's gold answer against the snapshot and reports the ones the
# snapshot cannot serve (run_benchmark's semantic oracle aborts the whole split on the first
# such task). With a snapshot built on 2026-10-09 that is exactly one task, analyze_ai5_ref4_desc:
# the 4th AI-pool paper of that day has no extractable figures.
CHECK_REPLAY = r'''
import json, os, sys
sys.path.insert(0, os.getcwd())
from tools.bootstrap import require_all_tools
require_all_tools()
from benchmark.tasks_expanded import get_expanded_tasks
from benchmark.semantic_oracle import attach_expected_paper_ids
split_file, split_name, snapshot = sys.argv[1:4]
tasks = {t["id"]: t for t in get_expanded_tasks()}
ids = json.load(open(split_file, encoding="utf-8"))["split"][split_name]
ok, bad = [], {}
for tid in ids:
    try:
        attach_expected_paper_ids([tasks[tid]], snapshot)
        ok.append(tid)
    except Exception as exc:
        bad[tid] = str(exc)[:300]
print("REPLAY_CHECK " + json.dumps({"ok": ok, "bad": bad}, ensure_ascii=False))
'''


def sh(cmd: list[str] | str, **kw) -> None:
    print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=str(REPO), check=True, shell=isinstance(cmd, str), **kw)
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)


def step(name: str):
    def wrap(fn):
        def run():
            mark = MARKS / name
            if mark.exists():
                print(f"[skip] {name}", flush=True)
                return
            print(f"\n===== {name} =====", flush=True)
            fn()
            mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        return run
    return wrap


def _restore_previous_output() -> None:
    """Resume from an attached Output of an earlier (failed) version of this notebook."""
    prev_marks = [p for p in Path("/kaggle/input").glob("**/marks") if p.is_dir() and INPUT not in p.parents]
    if not prev_marks:
        return
    prev = prev_marks[0].parent
    print(f"resuming from previous output: {prev}", flush=True)
    MARKS.mkdir(exist_ok=True)
    for mark in prev_marks[0].iterdir():
        if mark.name != "00_setup":
            shutil.copy(mark, MARKS / mark.name)
    for rel in ("eval", "artifacts"):
        src = prev / "AgenticArXiv-RL" / rel
        if src.exists():
            shutil.copytree(src, REPO / rel, dirs_exist_ok=True)
    for final in prev.glob("outputs/*/final"):
        dst = OUTPUTS / final.parent.name / "final"
        shutil.copytree(final, dst, dirs_exist_ok=True)
        print(f"  restored model {final.parent.name}", flush=True)
    # A marker is only trusted when the artifact it stands for is actually here.
    needed = {
        "10_train_round0": OUTPUTS / "round0" / "final" / "config.json",
        "30_train_round1": OUTPUTS / "round1" / "final" / "config.json",
        "20_self_evolve_run": REPO / "artifacts/self_evolve/round_1/targeted_parametric_seed.jsonl",
        "21_augment_and_mix": REPO / "artifacts/self_evolve/round_1/train_mix.jsonl",
    }
    for name, path in needed.items():
        if (MARKS / name).exists() and not path.exists():
            (MARKS / name).unlink()
            print(f"  dropped marker {name}: {path} missing", flush=True)
    # evaluate() skips splits whose summary.json exists, so re-running the eval steps is cheap.
    for name in ("11_eval_round0", "31_eval_round1", "40_gate", "50_collect"):
        (MARKS / name).unlink(missing_ok=True)


@step("00_setup")
def setup() -> None:
    MARKS.mkdir(exist_ok=True)
    if not REPO.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
             "https://github.com/hazelxue/AgenticArXiv-RL.git", str(REPO)],
            check=True,
        )
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "AgenticArxiv/requirements.txt"])
    (REPO / "data" / "sft").mkdir(parents=True, exist_ok=True)
    for name in ("mock_arxiv_snapshot.json",):
        shutil.copy(INPUT / name, REPO / "data" / name)
    sft_inputs = (
        "sft_v3_train_mix.jsonl",
        "sft_v3_train_mix.jsonl.manifest.json",
        "sft_v2_parametric_seed.jsonl.manifest.json",
    )
    for name in sft_inputs:
        shutil.copy(INPUT / name, REPO / "data" / "sft" / name)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (WORK / "check_replay.py").write_text(CHECK_REPLAY, encoding="utf-8")
    sh("df -h /kaggle/working /kaggle/temp")
    sh([sys.executable, "-c",
        "import torch, transformers, trl; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), "
        "'bf16', torch.cuda.is_bf16_supported(), 'torch', torch.__version__, 'transformers', transformers.__version__, "
        "'trl', trl.__version__)"])
    _restore_previous_output()


def train(data: str, name: str) -> None:
    out = OUTPUTS / name
    sh([sys.executable, "-m", "AgenticArxiv.rl.train_sft", "--model", BASE_MODEL, "--data", data,
        "--output_dir", str(out), *TRAIN_ARGS])
    # Keep the final model in /kaggle/working so a failed later step can resume without retraining.
    shutil.copytree(out / "final", SAVED_MODELS / name / "final", dirs_exist_ok=True)


def replayable_tasks(split: str) -> tuple[list[str], dict[str, str]]:
    proc = subprocess.run(
        [sys.executable, str(WORK / "check_replay.py"), f"../{SPLIT}", split, f"../{SNAPSHOT}"],
        cwd=str(REPO / "AgenticArxiv"), capture_output=True, text=True, check=True,
    )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("REPLAY_CHECK "))
    payload = json.loads(line[len("REPLAY_CHECK "):])
    return payload["ok"], payload["bad"]


def evaluate(model_dir: str, out_root: str) -> None:
    for split in EVAL_SPLITS:
        out = f"{out_root}/{split}"
        if (REPO / out / "summary.json").exists():
            continue
        ok, bad = replayable_tasks(split)
        (REPO / out).mkdir(parents=True, exist_ok=True)
        (REPO / out / "excluded_tasks.json").write_text(
            json.dumps({"split": f"{SPLIT}:{split}", "evaluated": ok, "excluded": bad},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if bad:
            print(f"  {split}: {len(bad)} task(s) not replayable with this snapshot, excluded: "
                  f"{sorted(bad)}", flush=True)
        selector = ["--split", f"{SPLIT}:{split}"] if not bad else ["--task-ids", *ok]
        sh([sys.executable, "-m", "AgenticArxiv.benchmark.run_benchmark", "--backend", "transformers",
            "--model", model_dir, "--local-device", "cuda", "--local-dtype", "float16", "--offline",
            "--snapshot", SNAPSHOT, "--task-set", "expanded", *selector,
            # Only the ReAct agent the SFT trains. run_benchmark defaults to regex+mcp+skill_cli, and
            # self_evolve groups summary rows by task_id only, so a multi-agent summary would mix
            # the mcp agent (needs the `mcp` package, absent here) and skill_cli into pass^3.
            "--agents", "regex",
            "--repeat", "3", "--seed", "45", "--output", out, "--save-traces", f"{out}/traces.jsonl"])


@step("10_train_round0")
def train_round0() -> None:
    train("data/sft/sft_v3_train_mix.jsonl", "round0")


@step("11_eval_round0")
def eval_round0() -> None:
    evaluate(str(OUTPUTS / "round0" / "final"), "eval/round0")


@step("20_self_evolve_run")
def self_evolve_run() -> None:
    sh([sys.executable, "-m", "AgenticArxiv.rl.self_evolve", "run", "--round", "1",
        "--eval", "dev=eval/round0/dev/summary.json", "--eval", "iid_test=eval/round0/iid_test/summary.json",
        "--traces", "eval/round0/dev/traces.jsonl", "--split", SPLIT, "--snapshot", SNAPSHOT,
        "--parametric-manifest", PARAMETRIC_MANIFEST, "--artifacts-dir", "artifacts/self_evolve"])


@step("21_augment_and_mix")
def augment_and_mix() -> None:
    rd = "artifacts/self_evolve/round_1"
    sh([sys.executable, "scripts/augment_parametric_sft_data.py", "--input", f"{rd}/targeted_parametric_seed.jsonl",
        "--split-file", SPLIT, "--wrapper-set", "targeted", "--output", f"{rd}/targeted_linguistic.jsonl"])
    sh([sys.executable, "-m", "AgenticArxiv.rl.self_evolve", "mix", "--base", "data/sft/sft_v3_train_mix.jsonl",
        "--targeted", f"{rd}/targeted_linguistic.jsonl", "--output", f"{rd}/train_mix.jsonl", "--round", "1"])


@step("30_train_round1")
def train_round1() -> None:
    train("artifacts/self_evolve/round_1/train_mix.jsonl", "round1")


@step("31_eval_round1")
def eval_round1() -> None:
    evaluate(str(OUTPUTS / "round1" / "final"), "eval/round1")


@step("40_gate")
def gate() -> None:
    def count_lines(path: Path) -> int:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0

    before = count_lines(REPO / "eval/eval_cases.jsonl")
    after = before + count_lines(REPO / "artifacts/self_evolve/round_1/eval_cases.open.jsonl")
    cmd = [sys.executable, "-m", "AgenticArxiv.rl.self_evolve", "gate", "--round-dir", "artifacts/self_evolve/round_1",
           "--split", SPLIT, "--cases-before", str(before), "--cases-after", str(after)]
    for split in EVAL_SPLITS:
        cmd += ["--prev", f"{split}=eval/round0/{split}/summary.json", "--new", f"{split}=eval/round1/{split}/summary.json"]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    print(f"gate exit code {rc} (0 = accepted, 1 = rejected; both are reported as-is)", flush=True)


@step("50_collect")
def collect() -> None:
    bundle = WORK / "round1_results"
    bundle.mkdir(exist_ok=True)
    shutil.copytree(REPO / "artifacts/self_evolve/round_1", bundle / "round_1", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("train_mix.jsonl", "targeted_linguistic.jsonl"))
    for rnd in ("round0", "round1"):
        for split in EVAL_SPLITS:
            src = REPO / f"eval/{rnd}/{split}"
            (bundle / rnd / split).mkdir(parents=True, exist_ok=True)
            for name in ("summary.json", "report.md", "excluded_tasks.json"):
                if (src / name).exists():
                    shutil.copy(src / name, bundle / rnd / split / name)
    for rnd in ("round0", "round1"):
        man = OUTPUTS / rnd / "final" / "training_manifest.json"
        if man.exists():
            shutil.copy(man, bundle / f"{rnd}_training_manifest.json")
    shutil.make_archive(str(WORK / "round1_results"), "gztar", root_dir=str(bundle))
    print("bundle:", WORK / "round1_results.tar.gz", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("STORE_BACKEND", "memory")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    steps = (
        setup, train_round0, eval_round0, self_evolve_run, augment_and_mix,
        train_round1, eval_round1, gate, collect,
    )
    for fn in steps:
        fn()
    print("\nROUND 1 COMPLETE", flush=True)
