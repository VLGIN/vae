import numpy as np
import torch
import argparse
import os

from src.model.vae import *
from src.data.data import *
from decouple import config
from loguru import logger
from utils import *

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cifar-10",
                        help="Path to data folder, which may contains one or several files of data.")
    parser.add_argument("--num_sample", type=int, default=32,
                        help="Number of sample to infer.")
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Checkpoint for infering.")

    arguments = parser.parse_args()

    data = read_data_from_disk(arguments.data_dir)
    data = data.reshape(config('IMG_SHAPE', cast=lambda x: [int(item) for item in x.split(",")]))

    num_img = np.arange(data.shape[0])
    sample_data = np.random.choice(num_img, arguments.num_sample)
    sample_data = data[sample_data]
    visualize_image(sample_data, "sample.png")

    image_shape = data.shape[1:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Load model
    model_path = os.path.join(config('MODEL_DIR'), f'checkpoint_epoch_{arguments.checkpoint}')
    logger.info(model_path)
    try:
        model = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.error(e)
        logger.info("Invalid model checkpoint")
        return
    
    model.to(device)
    model.eval()

    input = torch.tensor(sample_data, dtype=torch.float32, device=device)
    output = model(input)

    out_image = output["output"].cpu().detach().numpy()

    visualize_image(out_image, "infer.png")


if __name__ == "__main__":
    infer()
