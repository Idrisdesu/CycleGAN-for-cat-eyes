from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HumanCatEyesDataset(Dataset):
    def __init__(self, root_human, root_cat, transform=None):
        self.root_human = root_human
        self.root_cat = root_cat
        self.transform = transform

        self.human_images = os.listdir(root_human)
        self.cat_images = os.listdir(root_cat)
        self.length_dataset = max(len(self.human_images), len(self.cat_images))
        self.human_len = len(self.human_images)
        self.cat_len = len(self.cat_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        human_img = self.human_images[index % self.human_len]
        cat_img = self.cat_images[index % self.cat_len]

        human_path = os.path.join(self.root_human, human_img)
        cat_path = os.path.join(self.root_cat, cat_img)

        human_img = np.array(Image.open(human_path).convert("RGB"))
        cat_img = np.array(Image.open(cat_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=human_img, image0=cat_img)
            human_img = augmentations["image"]
            cat_img = augmentations["image0"]

        return human_img, cat_img




