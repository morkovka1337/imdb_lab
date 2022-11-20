import json
import os

import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import *
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
    def __init__(self, data_path, img_transforms=Compose([ToTensor()])):
        super().__init__()
        self.data_path = data_path
        self.img_transforms = img_transforms
        self.meta_info = pd.read_csv(os.path.join(data_path, 'movies_compact.csv'))

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, index):
        image_path = os.path.join(
            self.data_path, 'img', f"{self.meta_info['id'][index]}.jpg")
        image = cv.imread(image_path)
        if image is None:
            new_idx = random.choice(range(self.__len__()))
            return self.__getitem__(new_idx)
        # assert image is not None, f'wrong path {image_path}'
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, IMG_SIZE)
        image = self.img_transforms(image)
        name = self.meta_info['title'][index]

        description = self.meta_info['short'][index]
        assert isinstance(description, str), description
        genres = self.meta_info[ALLOWED_GENRES].iloc[index].tolist()
        return image, name, description, genres

class MMIMDBDataset(Dataset):
    def __init__(self, data_path, split='train', img_transforms=Compose([ToTensor()])):
        super().__init__()
        self.data_path = data_path
        with open(os.path.join(data_path, 'split.json')) as split_path:
            self.split_file = json.load(split_path)[split]
        image_paths = os.listdir(os.path.join(data_path, 'dataset'))
        meta_paths = os.listdir(os.path.join(data_path, 'dataset'))
        self.image_paths = [path for path in image_paths if path.split('.')[0] in self.split_file and path.endswith('.jpeg')]
        self.meta_paths = [path for path in meta_paths if path.split('.')[0] in self.split_file and path.endswith('.json')]
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.data_path, 'dataset', self.image_paths[idx])
        meta_path = image_path.replace('.jpeg', '.json')
        image = cv.imread(image_path)
        assert image is not None, f'wrong path {image_path}'
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, IMG_SIZE)
        image = self.img_transforms(image)

        with open(os.path.join(meta_path)) as meta_input:
            meta_info = json.load(meta_input)

        name = meta_info['title']
        description = meta_info['plot'][0][:MAX_DESCR_LEN]
        genres = meta_info['genres']
        for g in genres:
            if g in ALLOWED_GENRES:
                return image, name, description, g

        # none of the genres are in allowed genres, need to reroll batch
        new_idx = random.choice(range(self.__len__))
        return self.__getitem__(new_idx)

