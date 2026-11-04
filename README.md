# FSDG-YOLO

This repository contains the FSDG-YOLO model for fire and smoke detection, training and evaluation scripts for FASDD-UAV and D-Fire, and training records for three FASDD-UAV seeds.

## Installation

The reference environment uses Python 3.10, PyTorch 2.7.1 and Ultralytics 8.4.27. For the CUDA 12.8 setup, run the following commands from the repository root.

```sh
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Model

The three modules can be imported individually from [modules/](modules/).

| Module | File | Role |
| --- | --- | --- |
| DSDM | [DSDM.py](modules/DSDM.py) | Downsamples backbone features using separate detail and context branches. |
| DGFM | [DGFM.py](modules/DGFM.py) | Fuses a current feature map with a context feature map. |
| FAGM | [FAGM.py](modules/FAGM.py) | Enhances the P2 feature before multi-scale prediction. |

The model is defined in [FSDG-YOLO.yaml](configs/FSDG-YOLO.yaml). The first DGFM uses Backbone P5 as context for P4. The second uses the enhanced P4 feature as context for P5.

## Data

Supply a dataset YAML with the dataset root, split paths and class names. For FASDD-UAV, copy the train, validation and test lists from [splits/FASDD-UAV/](splits/FASDD-UAV/) to the dataset root alongside the images and labels. The class order is fire, smoke. For D-Fire, use its supplied partition and the class order smoke, fire.

## Training

[train.py](train.py) loads the dataset-specific settings from [configs/](configs/). FASDD-UAV training starts from the included YOLO11n weights. D-Fire training starts from scratch.

```sh
python train.py --dataset fasdd --data /path/to/fasdd.yaml --seed 15
python train.py --dataset dfire --data /path/to/dfire.yaml --seed 15
```

Use `--seed 7` or `--seed 57` for the additional FASDD-UAV runs. To check the model on CPU, run `python train.py --check`.

## Evaluation

[validate_ap_metrics.py](validate_ap_metrics.py) loads the custom modules and evaluates a checkpoint. It reports precision, recall, mAP50, mAP50–95, fire and smoke AP50, and AP-Small. To evaluate the included seed-15 checkpoint, run:

```sh
python validate_ap_metrics.py --model weights/seed15_best.pt --data /path/to/fasdd.yaml --split val --imgsz 640 --batch 16 --device 0
```

## Experiment records

The [seed 7](results/seed7/), [seed 15](results/seed15/) and [seed 57](results/seed57/) directories contain training arguments, per-epoch results and validation records. [seed_results.csv](results/seed_results.csv) collects their validation metrics as percentages.

The license is provided in [LICENSE](LICENSE).
