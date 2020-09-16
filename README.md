## Configuration
Copy file config.default.json to config.default and modify as necessary

## Training
> python train.py

The model will be trained and check points will be saved in `check_points/<timestamp_episode_x>`
## Submission
To prepare submission for Codalab, run
> python prepare_submission.py --chk_points/<timestamp_episode_x>
