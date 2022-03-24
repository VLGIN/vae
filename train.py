import numpy as np
import torch
import pickle
import argparse
import os
import sys

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.model.vae import *
from src.data.data import *
from decouple import config
from loguru import logger
from utils import *

def create_dataset(data, train_test_splits):
    data = np.array(data)
    ids = list(range(data.shape[0]))
    train_ids = np.random.choice(ids, size=len(ids)*(1-train_test_splits), replace=False)
    valid_ids = [item for item in ids if item not in train_ids]
    return Vae_Dataset(torch.tensor(data[train_ids], dtype=torch.float)), Vae_Dataset(torch.tensor(data[valid_ids], dtype=torch.float))

def create_dataloader(train_dataset, valid_dataset, arguments):
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=arguments.batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=arguments.batch_size)
    return train_dataloader, valid_dataloader

def validation(model, valid_dataloader, device):
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(valid_dataloader):
        image = batch
        image = image.to(device)
        output = model(image)

        total_loss += output["loss"].item()

    total_loss = total_loss / len(valid_dataloader)
    model.train()
    return total_loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cifar-10",
                        help="Path to data folder, which may contains one or several files of data.")
    parser.add_argument("--k_fold", type=int, default=9,
                        help="Numbef of folds to train with k-fold cross validation style, if k_fold=0, training with normal style.")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Max epoch number.")
    parser.add_argument("--num_cnn", type=int, default=3,
                        help="Numbef of CNN layer.")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dim of z.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--train_test_split", type=float, default=0.2,
                        help="Use when training in normal style.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for creating dataloader.")
    parser.add_argument("--train_from_checkpoint", action="store_true",
                        help="Continue training from checkpoint")

    arguments = parser.parse_args()

    if not os.path.exists(config('MODEL_DIR')):
        os.makedirs(config('MODEL_DIR'))

    data = read_data_from_disk(arguments.data_dir)
    data = data.reshape((-1, 3, 32, 32))
    if arguments.k_fold == 0:
        train_dataset, valid_dataset = create_dataset(data, arguments.train_test_split)
        train_dataloader, valid_dataloader = create_dataloader(train_dataset, valid_dataset, arguments)
    else:
        dataset = Vae_Dataset(torch.tensor(data, dtype=torch.float))
        kfold = KFold(arguments.k_fold)


    if arguments.train_from_checkpoint:
        logger.info("Load model from checkpoint.")
        hyper_param = process_log()
        current_best_loss = hyper_param["current_best_loss"]
        number_from_improvement = hyper_param["number_from_improvement"]
        current_epoch = hyper_param["current_epoch"]
        
        model = torch.load(os.path.join(config('MODEL_DIR'), 'checkpoint_epoch_{}'.format(current_epoch)))
    else:
        current_best_loss = float('inf')
        number_from_improvement = 0
        current_epoch = 0

        model = VAE(image_shape, arguments.num_cnn, arguments.latent_dim)

    image_shape = data.shape[1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(current_epoch+1, arguments.epoch):
        model.train()

        print(f"Training epoch {epoch}")
        logger.info("Training epoch {}".format(epoch))
        loss_valid_epoch = 0.0
        if arguments.k_fold != 0:
            logger.info("Training with k-fold cross validation style, numbef of folds: {}".format(arguments.k_fold))
            for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
                logger.info("Fold: {}".format(fold))
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

                train_dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size = arguments.batch_size,
                    sampler = train_subsampler
                )
                valid_dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size = arguments.batch_size,
                    sampler = valid_subsampler
                )
                
                total_loss = 0.0
                for i, batch in enumerate(train_dataloader):
                    image = batch
                    image = image.to(device)

                    optimizer.zero_grad()
                    output = model(image)
                    loss = output["loss"]
                    out_image = output["output"]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                    if i % 50 == 0 or i == len(train_dataloader) - 1:
                        logger.info("Batch {}/{}: loss {}({})".format(i+1, len(train_dataloader), loss.item(), total_loss / (i+1)))
                
                loss_valid = validation(model, valid_dataloader, device)
                loss_valid_epoch += loss_valid
                logger.info("Fold {}: loss {}".format(fold, loss_valid))
            loss_valid_epoch = loss_valid_epoch / arguments.k_fold
            logger.info("EPOCH {}: loss {}".format(epoch, loss_valid_epoch))            
        else:
            for i, batch in enumerate(train_dataloader):
                image = batch
                image = image.to(device)
                
                optimizer.zero_grad()
                output = model(image)
                loss = output["loss"]
                out_image = output["output"]
                loss.backward()
                optimizer.step()
                if i % 50 == 0 or i == len(train_dataloader) - 1:
                    logger.info("Batch {}/{}: loss {}({})".format(i+1, len(train_dataloader), loss.item(), total_loss / (i+1)))

            loss_valid_epoch = validation(model, valid_dataloader, device)
            logger.info("EPOCH {}: loss {}".format(epoch, loss_valid_epoch))

        model_path = os.path.join(config('MODEL_DIR'), "checkpoint_epoch_{}".format(epoch))
        torch.save(model, model_path)

        if loss_valid_epoch < current_best_loss:
            current_best_loss = loss_valid_epoch
            number_from_improvement = 0
        else:
            number_from_improvement += 1
        print(f"Epoch {epoch}, loss {loss_valid_epoch}, number from improvement {number_from_improvement}")
        if number_from_improvement >= 8:
            logger.info("TRAING END DUE TO POOR IMPROVEMENT ON VALIDATION DATA.")
            logger.info("BEST EPOCH {}".format(epoch - number_from_improvement))
            break
        
        if epoch == arguments.epoch - 1:
            logger.info("BEST EPOCH {}".format(epoch - number_from_improvement))
    

if __name__ == "__main__":
    logger.remove()
    #logger.add(sys.stderr, level="INFO")
    logger.add("training.log", level="INFO")
    train()