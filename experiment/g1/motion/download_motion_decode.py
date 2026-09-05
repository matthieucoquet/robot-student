import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

RETRY_DELAY_SECONDS = 5 * 60 + 1
DOWNLOAD_DIRECTORY = Path(__file__).parents[1] / "dataset" / "motion_decode"

while True:
    try:
        dataset_path = snapshot_download(
            repo_id="CMRobot/MotionDecode",
            repo_type="dataset",
            allow_patterns=[
                "samples/1.3.Basic_Gait_Category/1.3.1.Normal_Walking/**",
                "samples/4.Martial_Arts/**",
                "samples/5.Dance/**",
                "samples/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/**",
            ],
            local_dir=DOWNLOAD_DIRECTORY,
            max_workers=1,
            token=True,
        )
        break
    except (ConnectionError, HfHubHTTPError) as error:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code != 429 and "429 Too Many Requests" not in str(error):
            raise
        print(f"Hugging Face rate limit reached; retrying in {RETRY_DELAY_SECONDS} seconds.")
        time.sleep(RETRY_DELAY_SECONDS)

print(f"Downloaded: {dataset_path}")
