## Configuration
```
1. Copy file config.default.json to config.default and modify as necessary
2. conda env create -f environment.yml
3. conda activate grid2op
4. pip install -r requirements.txt
5. ./run_train.sh
```


## Training
> python train.py

The model will be trained and check points will be saved in `check_points/<timestamp_episode_x>`
## Submission
To prepare submission for Codalab, run
> python prepare_submission.py --chk_points/<timestamp_episode_x>
