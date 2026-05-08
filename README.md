# Robotics Hand Tracking — Data Labeling Pipeline

This is my internship project at Perspective Intelligence. The goal is to build a pipeline that automatically annotates hand-object interaction videos so we can use them as training data for robot manipulation tasks.

## What this does

Instead of manually watching hours of video and writing down what's happening (which is what I was doing before), these scripts automatically detect:
- where hands are in the video
- what objects are being touched
- whether the interaction is bimanual (both hands) or occluded (hand hidden behind object)

The output is a clean CSV that can be fed into a robot learning model.

## Project structure

```
data_labeling/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── hand_tracker_v2.py           # main tracker — run this on a video
│   ├── prepare_training_data.py     # cleans the Excel annotations into ML-ready CSV
│   ├── hand_data_extractor.py       # pulls MediaPipe keypoints from clips
│   ├── interaction_extractor.py     # pulls interaction events from video
│   ├── object_data_extractor.py     # pulls YOLO object detections
│   ├── advanced_video_registry.py   # manages the video file registry
│   └── make_registry.py             # builds metadata from folder structure
├── metadata/
│   └── training_dataset_clean.csv   # cleaned labeled dataset (458 segments)
└── Clip_Segmentation/
    └── final_clips/                 # segmented clips organized by task
```

## Setup

```bash
git clone https://github.com/perspective-intelligence/data_labeling.git
cd data_labeling
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run

### Hand tracker
```bash
python src/hand_tracker_v2.py
```
Change the `video_path` at the top of the file to point to your video. It will open a window showing the tracking live and save two files when done:
- a CSV with every contact event logged
- an annotated .mp4 video

### Prepare training data
```bash
python src/prepare_training_data.py
```
Put your Excel annotation file in the project root as `data.xlsx` first. Outputs a clean CSV to `metadata/training_dataset_clean.csv`.

## Tools used

- **MediaPipe** — hand detection, gives 21 keypoints (x,y,z) per hand per frame
- **YOLOv8-World** — object detection, draws bounding boxes around items
- **OpenCV** — video processing and drawing
- **pandas** — data cleaning

## Dataset

I manually labeled 458 video segments across 3 domains (grocery store, fruit picking, indoor tasks). Each segment has a task label, occlusion flag, bimanual flag, grip type, and material type. This is the ground truth the model will train on.

| | |
|---|---|
| Total segments | 458 |
| Task labels | 61 unique tasks |
| Occluded | 37% of segments |
| Bimanual | 50% of segments |

## Related papers I referenced

- [EgoZero](https://arxiv.org/pdf/2505.20290) — learning robot tasks from egocentric video with zero robot data
- [EgoMimic](https://arxiv.org/pdf/2410.24221) — co-training on human video + robot teleoperation data
- [Precise Affordances](https://arxiv.org/pdf/2408.10123) — teaching robots the difference between where to grasp vs what part is functional

## Next steps

- [ ] Write keypoint extraction script to pull x,y,z coordinates from each segmented clip
- [ ] Train a task classifier on the keypoint features
- [ ] Train an occlusion/bimanual detector to replace the current rule-based approach
