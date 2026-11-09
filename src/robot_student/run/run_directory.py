from datetime import datetime
from pathlib import Path


def find_run_directory(*, experiment_name: str, run_name: str, run_id: str) -> Path:
    result_directory = Path.cwd() / "result" / experiment_name
    if result_directory.exists():
        for directory in result_directory.iterdir():
            if directory.is_dir() and directory.name.endswith(run_id):
                return directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_directory = result_directory / f"{timestamp}_{run_name}_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory
