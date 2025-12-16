import modal
import os
import subprocess
import time
from pathlib import Path
import sys


def get_flag_value_from_list(lst, flag, default=None):
    if flag in lst:
        i = lst.index(flag)
        if i + 1 < len(lst):
            return lst[i + 1]
    return default


def args_to_string():
    args = sys.argv[5:]  # Skip: script, modal, run, --detach, train_remotely.py
    pairs = []
    
    i = 0
    while i < len(args):
        if args[i].startswith('--') and i + 1 < len(args) and not args[i + 1].startswith('--'):
            key = args[i][2:]
            value = args[i + 1]
            pairs.append(f"{key}={value}")
            i += 2
        else:
            i += 1
    
    return ";".join(pairs)


if modal.is_local():
    MODAL_EXP_ID = get_flag_value_from_list(sys.argv, "--exp_id")
    ARGS = args_to_string()
    if MODAL_EXP_ID is None:
        MODAL_EXP_ID = time.strftime("%Y_%m_%d_%H_%M_%S")
    APP_NAME = f"{ARGS}_{MODAL_EXP_ID}"
else:
    APP_NAME = ""

SRC_DIR = "."
MOUNT_AT = "/workspace"
DUMP_MOUNT_AT = "/dumped"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("dumped", create_if_missing=True)

image = (
    modal.Image.micromamba(python_version="3.13")
    .micromamba_install(
        "numpy",
        channels=["conda-forge"],
    )
    .pip_install("torch==2.9.0+cu128", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numba")
    .pip_install("psutil")
    .env({"PYTHONPATH": MOUNT_AT})
    .workdir(MOUNT_AT)
    .add_local_dir(local_path=".", remote_path="/workspace", ignore = ["checkpoint/**"])
)


@app.function(
    image=image,
    gpu="L4",
    timeout=24 * 60 * 60,
    volumes={DUMP_MOUNT_AT: vol},
)
def train(args = None):
    import shlex
    args = args or []
    vol.reload()
    os.makedirs(DUMP_MOUNT_AT, exist_ok=True)

    exp_name = get_flag_value_from_list(args, "--exp_name", "debug")
    exp_id = get_flag_value_from_list(args, "--exp_id")

    if exp_id is None:
        raise ValueError("exp_id is required")

    run_dir = Path(DUMP_MOUNT_AT) / exp_name / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MODAL_EXP_ID"] = exp_id

    has_dump_path = any(a.startswith("--dump_path") for a in args)
    cmd = ["python", "-u", "fc_loop.py", *args]
    if not has_dump_path:
        cmd += ["--dump_path", DUMP_MOUNT_AT]
    print("Launching:", " ".join(shlex.quote(a) for a in cmd))

    proc = subprocess.Popen(cmd, cwd=MOUNT_AT)
    ret = proc.wait()
    vol.commit()

    if ret != 0:
        raise SystemExit(ret)
    print("Training run completed.")


# modal run --detach train_remotely.py --env_name square --exp_name test_speed_new_code --N 40 --encoding_tokens single_integer --max_len 150 --temperature 0.6 --inc_temp 0.1 --shuffle true
@app.local_entrypoint()
def main(*args: str):
    args = list(args)
    if "--exp_id" not in args:
        args += ["--exp_id", MODAL_EXP_ID]
    train.remote(args)
