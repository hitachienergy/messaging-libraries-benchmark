from bml_optimizer.utils.logger import get_logger

from skopt import load
import subprocess as sp
import os
import glob

logger = get_logger(__name__)

def set_permissions(script_path):
    sp.run(["chmod", "+x", script_path], check=True)
    sp.run(["sudo", "setcap", "cap_net_bind_service=+ep", script_path], check=True)

def setup():
    script_path = os.path.join(os.path.dirname(__file__), "setup.sh")
    set_permissions(script_path)
    result = sp.run([script_path], capture_output=True, text=True)
    
def cleanup():
    script_path = os.path.join(os.path.dirname(__file__), "cleanup.sh")
    set_permissions(script_path)
    result = sp.run([script_path], capture_output=True, text=True)


def get_latest_file(directory: str, extension: str) -> str:
    files = glob.glob(f"{directory}/*.{extension}")
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def print_latest_opt_result() -> None:
    """
    Print the latest optimization result.
    """
    dump_dir = "./results/optimize_dump"
    latest_file = get_latest_file(dump_dir, "pkl")
    result = load(latest_file)
    print(f"Latest optimization result from {latest_file}:")
    print(result)