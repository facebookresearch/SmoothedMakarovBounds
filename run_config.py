"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import subprocess
import sys
from pathlib import Path

exp_id = "0"
config_files = ["config_synthetic"]
CONFIG_DIR = Path("configs")
PYTHON = sys.executable  # use the same Python interpreter
COMMAND_TEMPLATE = (
    '{python} synthetic_exp.py --config "{config_file}" --exp-id {exp_id}'
)
processes = []
for config_file in config_files:
    config_file_path = CONFIG_DIR / f"{config_file}.json"
    command = COMMAND_TEMPLATE.format(
        python=PYTHON,
        config_file=config_file_path,
        exp_id=exp_id,
    )
    print(f"Running flow for {config_file} -> {config_file_path}")
    # Prefer shell=False for safety; pass a string only if shell=True
    p = subprocess.Popen(command, shell=True)  # or: subprocess.Popen(command.split())
    processes.append(p)
