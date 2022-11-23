import json
import os

import cv2 as cv
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import pandas as pd


IMG_SIZE = [224, 224]
MAX_DESCR_LEN = 2048
ALLOWED_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Short",
    "Sport",
    "Superhero",
    "Thriller",
    "War",
    "Western",
]


class MMIMDBDatasetNew(Dataset):
    def __init__(self, data_path, split='train', img_transforms=None):
        super().__init__()
        self.data_path = data_path
        self.img_transforms = img_transforms
        self.meta_info = pd.read_csv(
            os.path.join(data_path, f'movies_{split}.csv'))

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, index):
        options = [
            os.path.join(
                self.data_path, 'img', f"{self.meta_info['id'][index]}.jpg"),
            os.path.join(
                self.data_path, 'img', f"0{self.meta_info['id'][index]}.jpg"),
            os.path.join(
                self.data_path, 'img', f"00{self.meta_info['id'][index]}.jpg"),
            os.path.join(
                self.data_path, 'img', f"000{self.meta_info['id'][index]}.jpg"),
        ]

        for file_path in options:
            if os.path.exists(file_path):
                image_path = file_path

        image = cv.imread(image_path)
        assert image is not None, f'wrong path {image_path}, {index=}'
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, IMG_SIZE)
        image = self.img_transforms(image=image)['image']
        name = self.meta_info['title'][index]

        description = self.meta_info['short'][index]
        assert isinstance(description, str), description
        genres = self.meta_info[ALLOWED_GENRES].iloc[index].tolist()
        return image, name, description, genres
