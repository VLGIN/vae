import numpy as np
import torch
import argparse
import os

from torch.utils.data import DataLoader
from src.model.vae import *
from src.data.data import *
from decouple import config
from loguru import logger
from utils import *

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="cifar-10-test",
                        help="Path to data folder, which may contains one or several files of data.")
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Checkpoint for infering.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generating images.")

    arguments = parser.parse_args()
    if not os.path.exists(arguments.infer_path):
        os.mkdir(arguments.infer_path)

    data = read_data_from_disk(arguments.data_dir)
    data = data.reshape(config('IMG_SHAPE', cast=lambda x: [int(item) for item in x.split(",")]))

    dataset = Vae_Dataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=arguments.batch_size)

    image_shape = config('IMG_SHAPE', cast=lambda x: [int(item) for item in x.split(",")])
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

    total_image = []
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        output = model(batch)
        images = output['output'].cpu().detach().numpy()
        total_image.append(images)

    total_image = np.concatenate(total_image, axis=0)
    fid = calculate_fid(data.reshape((data.shape[0], -1)), total_image.reshape(total_image.shape[0], -1))
    print(f"FID SCORE: {fid}")


if __name__ == "__main__":
    generate()
