# dataset cafe import
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2


class CustomDataset(Dataset):
    def __init__(self, images, images_identity, labels, labels_identity, input_size, base_transform, aug_transform, augment=False):
        self.images = images
        self.images_identity = images_identity
        self.labels = labels
        self.labels_identity = labels_identity
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.input_size == (64, 40):
            img = Image.fromarray(cv2.imread(self.images[idx], 0))
            img_identity = Image.fromarray(cv2.imread(self.images_identity[idx], 0))
        elif self.input_size == (40, 40):
            img = Image.fromarray(cv2.imread(self.images[idx], 0)[15:-9, :])
            img_identity = Image.fromarray(cv2.imread(self.images_identity[idx], 0)[15:-9, :])
        elif self.input_size == (48, 40):
            img = Image.fromarray(cv2.imread(self.images[idx], 0)[7:-9, :])
            img_identity = Image.fromarray(cv2.imread(self.images_identity[idx], 0)[7:-9, :])
        else: 
            raise NotImplemented
        # imread grayscale picture as array, PIL Image turn it into image

        if self.augment:
            base_img = self.base_transform(img)
            aug_img = self.aug_transform(img)
            label = torch.tensor(self.labels[idx]).type(torch.long)
            label_identity = torch.tensor(self.labels_identity[idx]).type(torch.long)

            return base_img, aug_img, label, label_identity

        elif not self.augment:
            base_img = self.base_transform(img)
            base_img_identity = self.base_transform(img_identity)
            label = torch.tensor(self.labels[idx]).type(torch.long)
            label_identity = torch.tensor(self.labels[idx]).type(torch.long)

            return base_img, base_img_identity, label, label_identity

        else:
            raise NotImplemented


def mapp(x):
    if x == 0:
        return 6
    elif x == 1:
        return 3
    elif x == 2:
        return 4
    elif x == 3:
        return 0
    elif x == 4:
        return 1
    elif x == 5:
        return 2
    elif x == 6:
        return 5
            
def read_label(filepath):
    """read label.csv and return a list"""

    file = open(filepath, 'r', encoding="utf-8")
    context = file.read()  # as str
    list_result = context.split("\n")[1:-1]
    length = len(list_result)

    for i in range(length):
        list_result[i] = int(list_result[i].split(",")[-1])

    file.close()  # file must be closed after manipulating
    
    return list(map(mapp,list_result))


def load_data(path, k_fold):

    filenames = [path + '/' + file for file in os.listdir(path) if os.path.splitext(file)[-1] == '.jpg']
    filenames.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))  
    # sorted by their order to match the label's order
    filenames_identity = ['./data/cafe/neutral_set' + '/' + file for file in os.listdir(path) 
                if os.path.splitext(file)[-1] == '.jpg']
    filenames_identity.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))  

    labels = read_label(os.path.join(path, 'label.csv'))
    labels_identity = read_label(os.path.join(path, 'label_identity.csv'))
    # read labels from the csv where labels have already been ordered.
    length = len(labels)

    if k_fold == None:
        return filenames, filenames_identity, labels, labels_identity
    elif "-" in k_fold:
        index = int(k_fold.split('/')[0][1:])
        folds = int(k_fold.split('/')[1])
        lower = int(length*(index-1)/folds)
        upper = int(length*(index)/folds)
        return filenames[:lower]+filenames[upper:], filenames_identity[:lower]+filenames_identity[upper:], labels[:lower]+labels[upper:], labels_identity[:lower]+labels_identity[upper:]
    else:
        index = int(k_fold.split('/')[0])
        folds = int(k_fold.split('/')[1])
        lower = int(length*(index-1)/folds)
        upper = int(length*(index)/folds)
        return filenames[lower:upper], filenames_identity[lower:upper], labels[lower:upper], labels_identity[lower:upper]


def get_cafeloaders(path='./data/cafe/balance_all', bs=64, augment=False, input_size=(64, 40), k_fold=None):
    """ Prepare dataloaders
        Augment training data using:
            - horizental flipping
            - rotation
        input: path to ft_dataset file
        output: Dataloaders
        k_fold: None or '1/10', '2/5', '-3/6', ..."""

    filenames, filenames_identity, labels, labels_identity = load_data(path, k_fold)

    base_transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize((48,48)),
        # transforms.FiveCrop(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(255,))
    ])

    if augment:
        aug_transform = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.RandomResizedCrop((75, 55), scale=(0.8, 1.2)),
            # transforms.RandomApply([transforms.ColorJitter(
            #     brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            # transforms.RandomApply(
            #     [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            # transforms.FiveCrop((64, 40)),
            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.Normalize(mean=(0,), std=(1,))(t) for t in tensors])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.RandomErasing()(t) for t in tensors])),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=(0,), std=(1,))
        ])
    else:
        aug_transform = base_transform

    dataset = CustomDataset(filenames, filenames_identity, labels, labels_identity, input_size, base_transform, aug_transform, augment)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)
    # use multi-thread to load the dataset

    return dataloader
