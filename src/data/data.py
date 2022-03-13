from torch import nn
from torch.utils.data import Dataset


class Vae_Dataset(Dataset):
    def __init__(self, images):
        super(Vae_Dataset, self).__init__()
        self.images = images
    
    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.images.shape[0]

class DataLoader():
    pass