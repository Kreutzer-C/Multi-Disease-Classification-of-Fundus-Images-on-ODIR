from torch.utils.data import Dataset
from PIL import Image
import os

class ODIRDataSet(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_path1 = os.path.join('./dataset/ODIR-5K_Training_Dataset', self.df.iloc[item, 0])
        img_path2 = os.path.join('./dataset/ODIR-5K_Training_Dataset', self.df.iloc[item, 1])
        labels = self.df.iloc[item, 2:].values.astype("float32")

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, labels