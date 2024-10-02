"""
Adapted from https://github.com/usef-kh/fer/tree/master/data
"""
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, desired_data='all'):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.desired_data = desired_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        # if self.desired_data == 'all':
        #     label = torch.tensor(self.labels[idx]).type(torch.long)
        # else:
        #     # turn the label 3, 4 into 0, 1
        #     label = torch.tensor(self.labels[idx] - min(self.desired_data)).type(torch.long)
        label = torch.tensor(self.labels[idx]).type(torch.long)

        return img, label


def load_data(path='datasets/fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    # print(image_array.shape)

    return image_array, image_label


def get_dataloaders(path='data/fer2013/fer2013.csv', desired_data='all', bs=64, augment=True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    if desired_data == 'all':
        xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
        xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
        xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])
    else:
        xtrain, ytrain = prepare_data(fer2013[(fer2013['Usage'] == 'Training') & (fer2013['emotion'].isin(desired_data))])
        xval, yval = prepare_data(fer2013[(fer2013['Usage'] == 'PrivateTest') & (fer2013['emotion'].isin(desired_data))])
        xtest, ytest = prepare_data(fer2013[(fer2013['Usage'] == 'PublicTest') & (fer2013['emotion'].isin(desired_data))])

    mu, st = 0, 255

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.FiveCrop(40),
        # transforms.Lambda(lambda crops: torch.stack(
        #     [transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda tensors: torch.stack(
        #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        # transforms.Resize((40,40)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(255,)), 
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.FiveCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors])),
        ])
    else:
        train_transform = test_transform

    # X = np.vstack((xtrain, xval))
    # Y = np.hstack((ytrain, yval))

    train = CustomDataset(xtrain, ytrain, train_transform, desired_data)
    val = CustomDataset(xval, yval, test_transform, desired_data)
    test = CustomDataset(xtest, ytest, test_transform, desired_data)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=bs, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=bs, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader
