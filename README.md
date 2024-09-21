
### About ###
*KM-MOD* provides a standalone image classifier for use with date-time stamped (YYYYMMDD-HHMMSS) images which most security cameras produce. Repetitive images are filtered out leaving only images of interest. Selected images are converted into a time-lapse video. 

### Requirements
* RPI4 2GB or higher
* Python 3.4 or higher

### Setup
```
python -m pip install -U pip
python -m pip install -U scikit-image 
pip install opencv-python
pip install shutils
pip install -U scikit-learn (for kmeans)
pip install matplotlib

sudo apt update
sudo apt upgrade
sudo apt install ffmpeg
```

### Config
Paths/variables need to be set to actual paths/values at the beginning of these files.
```
config.py
daily-driver-mp
moviefrm-list
moviefrm-list-ni (non-interactive version used by `daily-driver-mp`)
```

### Getting Started
`git clone https://github.com/SensorAnalyticsAus/KM-MOD.git`
`cd KM-MOD`

### Example 1
Step 1 train

`/path/to/.venv/bin/python train-km-mp.py on 1 10 20240506000000 20240506235959`

Step 2 predict (output frames from selected clusters from step 1)

`/path/to/.venv/bin/python predict-km.py 80`

Step 3 create a time-lapse video

`./moviefrm-list 20`

### Example 2
`./daily-driver-mp on` 

or

`./daily-driver-mp off` 

`off` prevents the progress-bar from showing, useful for `cron`
This is best run at say 7am and 7pm daily as a `cron` job