# Mutual Learning to Adapt for Joint Human Parsing and Pose Estimation

This repository contains the code and pretrained models of
> **Mutual Learning to Adapt for Joint Human Parsing and Pose Estimation** [[PDF](https://niexc.github.io/assets/pdf/MuLA_eccv2018.pdf)]   
> **Xuecheng Nie**, Jiashi Feng, and Shuicheng Yan   
> European Conference on Computer Vision (ECCV), 2018    

## Prerequisites

- Python 3.5
- Pytorch 0.2.0
- OpenCV 3.0 or higher

## Installation

1. Install Pytorch: Please follow the [official instruction](https://pytorch.org/) on installation of Pytorch.
2. Clone the repository   
   ```
   git clone --recursive https://github.com/NieXC/pytorch-pil.git
   ``` 
3. Download [Look into Person (LIP)](http://sysu-hcp.net/lip/overview.php) dataset and create symbolic links to the following directories
   ```
   ln -s PATH_TO_LIP_TRAIN_IMAGES_DIR dataset/lip/train_images   
   ln -s PATH_TO_LIP_VAL_IMAGES_DIR dataset/lip/val_images      
   ln -s PATH_TO_LIP_TEST_IMAGES_DIR dataset/lip/testing_images   
   ln -s PATH_TO_LIP_TRAIN_SEGMENTATION_ANNO_DIR dataset/lip/train_segmentations   
   ln -s PATH_TO_LIP_VAL_SEGMENTATION_ANNO_DIR dataset/lip/val_segmentations   
   ```

## Usage

### Training
Run the following command to train the model from scratch (Default: 5-stage Hourglass based network):
```
sh run_train.sh
```
or 
```
CUDA_VISIBLE_DEVICES=0,1 python main.py -b 24 --lr 0.0015
```

A simple way to record the training log by adding the following command:
```
2>&1 | tee exps/logs/mula_lip.log
```

Some configurable parameters in training phase:

- `--arch` network architecture (HG (Hourglass) or VGG)
- `-b` mini-batch size   
- `--lr` initial learning rate (0.0015 for HG based model and 0.0001 for VGG based model)
- `--epochs` total number of epochs for training
- `--snapshot-fname-prefix` prefix of file name for snapshot, e.g. if set '--snapshot-fname-prefix exps/snapshots/mula_lip', then 'mula_lip.pth.tar' (latest model), 'mula_lip_pose_best.pth.tar' (model with best validation PCK for human pose estimation) and 'mula_lip_parsing_best.pth.tar' (model with best validation mIoU for human parsing) will be generated in the folder 'exps/snapshots' 
- `--resume` path to the model for recovering training
- `-j` number of workers for loading data
- `--print-freq` print frequency

### Testing
Run the following command to evaluate the model on LIP `validation set`:
```
sh run_test.sh
```
or 
```
CUDA_VISIBLE_DEVICES=0 python main.py --evaluate True --calc-pck True --calc-miou True --resume exps/snapshots/mula_lip.pth.tar
```

Run the following command to evaluate the model on LIP `testing set`:
```
CUDA_VISIBLE_DEVICES=0 python main.py --evaluate True --resume exps/snapshots/mula_lip.pth.tar --eval-data dataset/lip/testing_images --eval-anno dataset/lip/jsons/LIP_SP_TEST_annotations.json
```

In particular, human pose estimation results will be saved as a `.csv` file followed the official evaluation format of LIP dataset for single-person human pose estimation. An example is provided in `exps/preds/csv_results/pred_keypoints_lip.csv`. Human parsing results will be saved as a set of `.png` images in the folder `exps/preds/parsing_results`, representing the segmentation label maps for testing images.

Some configurable parameters in testing phase:

- `--evaluate` True for testing and false for training
- `--resume` path to the model for evaluation
- `--calc-pck` calculate PCK or not for validation of human pose estimation
- `--calc-miou` calculate mIoU or not for validation of human parsing
- `--pose-pred-path` path to the csv file for saving the human pose estimation results
- `--parsing-pred-dir` directory to the png images for saving the human parsing results
- `--visualization` visualize evaluation or not
- `--vis-dir` directory for saving the visualization results

## Citation

If you use our code in your work or find it is helpful, please cite the paper:
```
@inproceedings{nie2018mula,
  title={Mutual Learning to Adapt for Joint Human Parsing and Pose Estimation},
  author={Nie, Xuecheng and Feng, Jiashi and Yan, Shuicheng},
  booktitle={ECCV},
  year={2018}
}
```
