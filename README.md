# Variational Auto Encoder Implemetation

Reference: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## How to train
```bash
python3 train.py --data_dir path_to_data_folder --epoch number_of_max_epoch --lr learning_rate
```
Config image shape in .env file

Use ```python3 train.py --help``` for all options training VAE model

## How to infer
```bash
python3 infer.py --data_dir path_to_data_folder --num_img number_of_image_for_infering --checkpoint which_checkpoint_to_be_used
```
Use ```python3 infer.py --help``` for all options.

## How to evaluation (calculate FID)
```bash
python3 eval.py --data_dir path_to_data_folder --checkpoint which_checkpoint_to_be_used --batch_size bach_size
```
Use ```python3 eval.py --help``` for all options.
