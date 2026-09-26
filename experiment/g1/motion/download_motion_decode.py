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
                "samples/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/**",
                "samples/1.2.State_Transition_Category/1.2.1.Speed_Transition_Still_Walk/**",
                "samples/1.2.State_Transition_Category/1.2.1.Direction_Switch_Straight_Turn/**",
                "samples/1.3.Basic_Gait_Category/1.3.1.Normal_Walking/**",
                "samples/1.3.Basic_Gait_Category/1.3.2.Fast_Walking_Jogging/1.3.2.2.Jogging/**",
                "samples/1.14.Dance_and_Performance/**",
                "samples/1.5.Balance_Control_Type/**",
                "samples/2.4.Competitive_Interaction/2.4.1.Box_Ing/2.4.1.2.Hook_Punch/**",
                "samples/4. Martial_Arts/**",
                "samples/5.Dance/**",
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
