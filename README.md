# RefTr: Recurrent Refinement of Confluent Trajectories for 3D Vascular Tree Centerline Graphs

This repository provides the official implementation of the RefTr: Recurrent Refinement of Confluent Trajectories for 3D Vascular Tree Centerline Graphs paper by [Roman Naeem](https://research.chalmers.se/en/person/nroman), [David Hagerman](https://research.chalmers.se/en/person/olzond), [Jennifer Alvén](https://research.chalmers.se/person/alven) and [Fredrik Kahl](https://research.chalmers.se/person/kahlf). 

### Note
We are currently in the process of cleaning up the code and updating the hyperparameters.

## Installation
1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Install PyTorch 2.2 with CUDA 11.8:
    ```
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    ```

<hr>

## Training

### Data Preparation
1. Download the desired [datasets](https://zenodo.org/records/15888958) to `./data`.
2. For ATM'22 and Parse 2022 only centerline graphs are provided. Download the images and segmentation masks from [ATM'22](https://atm22.grand-challenge.org/) and [Parse 2022](https://parse2022.grand-challenge.org/) and resample them to 0.5 mm isotropically.
3. Run the `./src/reftr/datasets/utils/organize_data.py` script to organize the data into the required directory structure.
4. Run the script `./src/reftr/datasets/utils/generate_val_sub_vol_file.py` to generate 'annots_val_sub_vol.pickle' file that contains the information about annotations for the validation sub-volume images. 
5. Running the scripts above should give you the following data directory structure:
    ```
    Main-Directory
    ├── data
    │   ├── dataset_name
    │   │   ├── annots_train
    │   │   ├── annots_val
    │   │   ├── annots_val_sub_vol
    │   │   ├── annots_test 
    │   │   ├── images_train
    │   │   ├── images_val
    │   │   ├── images_val_sub_vol
    │   │   ├── images_test
    │   │   ├── masks_train
    │   │   ├── masks_val
    │   │   ├── masks_val_sub_vol
    │   │   ├── masks_test
    │   │   ├── annots_val_sub_vol.pickle
    ```
   The '_train', '_val', and '_test' directories contain the training, validation, and test images respectively. The '_val_sub_vol' directory contains the validation images that are used for patch-level evaluation. The 'masks' directories contain the binary masks of the vessel trees. The 'annots' directories contain the annotation files in the required format.

### Training
The training script uses the configuration file `./configs/train.yaml` to set the hyperparameters. To train the model, run the following command from the root directory:
```
python ./src/train.py
```
For distributed training, use the following command:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS ./src/train.py
```

<hr>

## Evaluation
For evaluation, in addition to `./configs/train.yaml`, we use  `./configs/eval.yaml`. To evaluate the model, run the following command from the root directory:
```
python ./src/train.py with eval
```
For distributed training, use the following command:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS ./src/train.py with eval
```

<hr>
