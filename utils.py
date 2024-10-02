# utils for vMF-VAE modelling
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, atan2, asin, sqrt
from torchvision.transforms import CenterCrop
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def latent_space(model, dataloader, config, legend=True, dataset="cafe"):
    """display how the latent space clusters different data points in vMF space"""

    model.eval()
    x = torch.rand((1, model.submodel2.z_dim))
    y = torch.rand(1)

    for data in dataloader:  # calculate the latent codes
        images, base_images_id, labels, labels_id = data
        
        images, labels= images.to(config.device), labels.to(config.device)

        if dataset == "cafe":
            latent_code, _, _, _ = model(CenterCrop((40, 40))(images))
        elif dataset == "fer2013":
            latent_code, _, _, _ = model(images)

        x = torch.cat((x, latent_code[0].to("cpu")), 0) 
        y = torch.cat((y, labels.to("cpu")), 0) 

    x = x.detach().numpy()[1:, :]
    y = y.detach().numpy()[1:]

    if x.shape[1] == 3:
        fig = plt.figure(figsize=(10, 15))

        ax = fig.add_subplot(111, projection='3d')
        # ax2 = fig.add_subplot(212)
        for key, value in config.label_list.items():
            x_, y_, z_ = x[np.where(y == int(key)), 0], x[np.where(y == int(key)), 1], x[np.where(y == int(key)), 2]
            ax.scatter(x_, y_, z_, label=value)

            if config.distribution_emo == "vmf":
                x__, y__ = [asin(i) for i in z_[0]], [sqrt(1-i[2]*i[2]) * atan2(i[0], i[1])
                for i in x[np.where(y == int(key)), :][0]]
                ax2.scatter(y__, x__, label=value)
                ax2.set_aspect('equal')

        # plt.title("Data Points in Latent Space")
        if legend == True:
            ax.legend(fontsize=12)
        # plt.savefig('./figure/save.jpg')
        # ax.set_xlim(-1 * config.radius, config.radius)
        # ax.set_ylim(-1 * config.radius, config.radius)

        # ax.set_xlabel("X", fontsize=12)
        # ax.set_ylabel("Y", fontsize=12)

        # ax.set_zlim(-1 * config.radius, config.radius)
        # ax.set_zlabel("Z", fontsize=12)
        # ax.tick_params(axis='both', which='major', labelsize=15)

    elif x.shape[1] == 2:
        fig, ax = plt.subplots()
        for key, value in config.label_list.items():
            x_, y_ = x[np.where(y == int(key)), 0], x[np.where(y == int(key)), 1]
            ax.scatter(x_, y_, label=value)
        if legend == True:
            ax.legend(fontsize=12)
    else: 
        raise NotImplemented

    print("Number of data:", len(x))

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def manifold_sphere(model, config, theta=(0,), z_resolution=10, z=0):
    """
    theta: the arc between intersecting lines - one is (0, 1) on the x-y plane, ranged in 0-2pi;
    z_resolution: how many pics to be shown in a 1/4 arc
    """
    model.eval()

    if model.z_dim == 3 and model.distribution == "vmf":
        fig, ax = plt.subplots(len(theta), z_resolution)
        plt.setp(ax.flat, xticks=[], yticks=[])

        for j, arc in enumerate(theta):
            grid = torch.tensor([[0, 0, 0]])

            for i in range(z_resolution):

                phi = pi*(-0.5) + (i+1)*pi/z_resolution
                sample = torch.tensor([[np.sin(arc)*np.cos(phi), np.cos(arc)*np.cos(phi), np.sin(phi)]]) * config.radius
                grid = torch.cat((grid, sample), 0)


            grid = grid[1:, :].type("torch.FloatTensor").to(config.device)
            img = model.decoder(grid)
            img = img.cpu().detach().numpy()

            for i in range(z_resolution):
                ax[j][i].imshow(img[i, 0, :, :],cmap='gray')

        fig.subplots_adjust(wspace=0, hspace=0)

    if model.z_dim == 3 and model.distribution == "normal":
        fig, ax = plt.subplots(z_resolution, z_resolution)
        plt.setp(ax.flat, xticks=[], yticks=[])

        for i in range(z_resolution):
            for j in range(z_resolution):
                p = torch.tensor([[-5 + 10*j / z_resolution, 5 - 10*i / z_resolution, z]]).type("torch.FloatTensor").to(config.device)
                sample = model.decoder(p)
                sample = sample.cpu().detach().numpy()
                ax[j][i].imshow(sample[0, 0, :, :],cmap='gray')

        fig.subplots_adjust(wspace=0, hspace=0)


    elif model.z_dim == 2:
        fig, ax = plt.subplots(z_resolution, z_resolution)
        plt.setp(ax.flat, xticks=[], yticks=[])

        for i in range(z_resolution):
            for j in range(z_resolution):
                p = torch.tensor([[-8 + 16 * j / z_resolution, 8 - 16 * i / z_resolution]]).type(
                    "torch.FloatTensor").to(config.device)
                sample = model.decoder(p)
                sample = sample.cpu().detach().numpy()
                ax[j][i].imshow(sample[0, 0, :, :], cmap='gray')

        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)

    else: 
        raise NotImplemented

    plt.tight_layout()
    plt.show()
    
    
def conditional_manifold(model, obj, condition, config, theta=(0,), z_resolution=10):
    """
    theta: the arc between intersecting lines - one is (0, 1) on the x-y plane, ranged in 0-2pi;
    z_resolution: how many pics to be shown in a 1/4 arc
    """
    model.eval()

    if model.z_dim_id == 3 and model.distribution_id == "vmf" and obj == "id":
        fig, ax = plt.subplots(len(theta), z_resolution)
        plt.setp(ax.flat, xticks=[], yticks=[])

        for j, arc in enumerate(theta):
            grid = torch.tensor([[0, 0, 0]])

            for i in range(z_resolution):

                phi = pi*(-0.5) + (i+1)*pi/z_resolution
                sample = torch.tensor([[np.sin(arc)*np.cos(phi), np.cos(arc)*np.cos(phi), np.sin(phi)]]) * config.radius
                grid = torch.cat((grid, sample), 0)

            padding = condition.repeat(len(grid), 1)
            grid = torch.cat((grid, padding), 1)

            grid = grid[1:, :].type("torch.FloatTensor").to(config.device)
            img = model.decoder(grid)
            img = img.cpu().detach().numpy()

            for i in range(z_resolution):
                ax[j][i].imshow(img[i, 0, :, :],cmap='gray')

        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)

    elif model.z_dim_emo == 3 and model.distribution_emo == "vmf" and obj == "emo":
        fig, ax = plt.subplots(len(theta), z_resolution)
        plt.setp(ax.flat, xticks=[], yticks=[])

        for j, arc in enumerate(theta):
            grid = torch.tensor([[0, 0, 0]])

            for i in range(z_resolution):

                phi = pi*(-0.5) + (i+1)*pi/z_resolution
                sample = torch.tensor([[np.sin(arc)*np.cos(phi), np.cos(arc)*np.cos(phi), np.sin(phi)]]) * config.radius
                grid = torch.cat((grid, sample), 0)

            padding = condition.repeat(len(grid), 1)
            grid = torch.cat((padding, grid), 1)

            grid = grid[1:, :].type("torch.FloatTensor").to(config.device)
            img = model.decoder(grid)
            img = img.cpu().detach().numpy()

            for i in range(z_resolution):
                ax[j][i].imshow(img[i, 0, :, :],cmap='gray')

        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)

    elif model.z_dim == 2:
        fig, ax = plt.subplots(1, len(theta))
        plt.setp(ax.flat, xticks=[], yticks=[])

        grid = torch.tensor([[0, 0]])

        for arc in theta:

            sample = torch.tensor([[np.sin(arc), np.cos(arc)]]) * config.radius
            grid = torch.cat((grid, sample), 0)

        grid = grid[1:, :].type("torch.FloatTensor").to(config.device)
        img = model.decoder(grid)
        img = img.cpu().detach().numpy()

        for i in range(len(theta)):
            ax[i].imshow(img[i, 0, :, :],cmap='gray')

    else: 
        raise NotImplemented

    plt.show()


def latent_sample(model, config, n_show=10, std=10):

    # randomly choose a batch to show
    img_show = torch.normal(0, std, size=(n_show, 3))
    model.eval()

    img_show = img_show.type("torch.FloatTensor").to(config.device)
    img = model.decoder(img_show)
    img = img.cpu().detach().numpy()

    # plt.style.use("dark_background")
    f, a = plt.subplots(1, n_show)
    for i in range(n_show):
        a[i].imshow(img[i, 0, :, :],cmap='gray')
    plt.show()


def reconstruction(model, dataloader, n_show, config):
    # random visualization of the reconstruction
    model.eval()

    rand_idx = np.random.randint(0, len(dataloader), 1)

    for idx, data in enumerate(dataloader):  # calculate the latent codes
        if idx == rand_idx:

            if config.aug:
                if config.Ncrop:
                    _, _, c, h, w = data[1].shape
                    images = data[1].to(config.device).view(-1, c, h, w)
                else:
                    images = data[1].to(config.device)
            elif not config.aug:
                images = data[0].to(config.device)

            _, _, _, (img_rec, _) = model(CenterCrop((40, 40))(images))

            f, a = plt.subplots(2, n_show)
            plt.setp(a.flat, xticks=[], yticks=[])

            for i in range(n_show):
                a[0][i].imshow(images.cpu().numpy()[i, 0, :, :],cmap='gray')
                a[1][i].imshow(img_rec.cpu().detach().numpy()[i, 0, :, :],cmap='gray')

            f.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)
            plt.show()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='earlystopping_checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.measure = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, measure, direction, model):

        if direction == "min":
            score = -measure
        elif direction == "max":
            score = measure

        if self.best_score is None:
            self.best_score = score
            self.save_all(measure, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_all(measure, model)
            self.counter = 0

    def save_checkpoint(self, measure, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation measure ({self.measure:.6f} --> {measure:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.measure = measure

    def save_all(self, measure, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation measure ({self.measure:.6f} --> {measure:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.measure = measure


def identity_test(model, dataloader, config):
    """to obtain how much the model have learned from the identity information of the faces"""
    model.eval()
    latent_code = torch.rand((1, model.z_dim))
    label_id = torch.rand(1)
    label_emo = torch.rand(1)

    for data in dataloader:  # calculate the latent codes
        images = data[0].to(config.device)
        (embeddings, _), _, z, _ = model(images)
        embeddings = embeddings.to("cpu")

        latent_code = torch.cat((latent_code, embeddings), 0) 
        label_emo = torch.cat((label_emo, data[-2]), 0) 
        label_id = torch.cat((label_id, data[-1]), 0) 

    latent_code = latent_code.detach().numpy()[1:, :]
    label_emo = label_emo.detach().numpy()[1:]
    label_id = label_id.detach().numpy()[1:]

    distance_within_p_across_emo = 0
    distance_between_p_across_emo = 0
    # within/between subject distance ratio 
    # for each emo cluster
    for i in range(7):

        latent_code_sub = latent_code[np.where(label_emo==i), :][0]
        label_id_sub = label_id[np.where(label_emo==i)]

        distance_within_p = 0
        ave_position_p = np.ones((1, model.z_dim))

        for j in np.unique(label_id_sub):
            # for each participant within each emo cluster
            idx = np.where(label_id_sub==j)

            if len(idx) > 1:
                distance_self = center_mean_distance(latent_code_sub[idx, :])
            else:
                distance_self = 0

            distance_within_p += distance_self
            mean_position = np.mean(latent_code_sub[idx, :][0], axis=0, keepdims=True)
            mean_position = mean_position / sqrt(np.sum(mean_position ** 2))
            ave_position_p = np.concatenate((ave_position_p, mean_position), 0)

        distance_within_p_across_emo += distance_within_p / len(np.unique(label_id_sub))
        distance_between_p_across_emo += pairwise_mean_distance(ave_position_p[1:, :])

    return distance_within_p_across_emo, distance_between_p_across_emo


def center_mean_distance(data):
    # data should be an array containing a bunch of coordinates
    dt = 0

    mean_position = np.mean(data, axis=0, keepdims=True)
    mean_position = mean_position / sqrt(np.sum(mean_position ** 2))

    for i in range(len(data)):
        d = sqrt(np.sum((data[i, :] - mean_position) ** 2))
        dt += 2 * np.arcsin(d / 2)

    return dt/len(data)


def pairwise_mean_distance(data):
    # data should be an array containing a bunch of coordinates
    dt = 0

    for i in range(len(data)-1):
        for j in range(i, len(data)):
            d = sqrt(np.sum((data[i, :] - data[j, :]) ** 2))
            dt += 2 * np.arcsin(d / 2)

    return dt * 2 / (len(data)*(len(data)-1))


def collapse_test(model, dataloader, config):
    """to obtain how much the model have learned from the identity information of the faces"""
    model.eval()
    latent_code = torch.rand((1, model.z_dim))
    label_emo = torch.rand(1)

    for data in dataloader:  # calculate the latent codes
        images = data[0].to(config.device)
        (embeddings, _), _, z, _ = model(images)
        embeddings = embeddings.to("cpu")

        latent_code = torch.cat((latent_code, embeddings), 0) 
        label_emo = torch.cat((label_emo, data[-2]), 0) 

    latent_code = latent_code.detach().numpy()[1:, :]
    label_emo = label_emo.detach().numpy()[1:]

    distance_within_emo = 0
    ave_position_emo = np.ones((1, model.z_dim))
    # within/between subject distance ratio 
    # for each emo cluster
    for i in range(7):

        latent_code_sub = latent_code[np.where(label_emo==i), :][0]
        distance_in = center_mean_distance(latent_code_sub)
        distance_within_emo += distance_in / 7

        mean_position = np.mean(latent_code_sub, axis=0, keepdims=True)
        mean_position = mean_position / sqrt(np.sum(mean_position ** 2))
        ave_position_emo = np.concatenate((ave_position_emo, mean_position), 0)

    distance_between_emo = pairwise_mean_distance(ave_position_emo[1:, :])

    return distance_between_emo / distance_within_emo

def accuracy_cafe(model, dataloader, config):
    model.eval()
    positive_pred = 0
    amount = 0

    for data in dataloader: 
        base_images, base_images_id, labels, labels_id = data
        
        base_images, labels= base_images.to(config.device), labels.to(config.device)
        # base_images_id, labels_id= base_images_id.to(config.device), labels_id.to(config.device)
        
        _, _, _, (_, labels_emo_) = model(CenterCrop((40, 40))(base_images))
        positive_pred += torch.sum(torch.argmax(labels_emo_, dim=1, keepdim=False) == labels)
        amount += len(labels)
    
    accuracy = positive_pred.cpu().numpy() / amount
    
    return accuracy


def accuracy_fer(model, dataloader, config):
    model.eval()
    positive_pred = 0
    amount = 0

    for data in dataloader: 
        base_images, labels = data
        
        base_images, labels= base_images.to(config.device), labels.to(config.device)
        # base_images_id, labels_id= base_images_id.to(config.device), labels_id.to(config.device)
        
        _, _, _, (_, labels_emo_) = model(base_images)
        positive_pred += torch.sum(torch.argmax(labels_emo_, dim=1, keepdim=False) == labels)
        amount += len(labels)
    
    accuracy = positive_pred.cpu().numpy() / amount
    
    return accuracy


def accuracy_res(model, dataloader, config, size):
    model.eval()
    positive_pred = 0
    amount = 0

    for data in dataloader: 
        try:
            base_images, labels = data
            base_images, labels= base_images.to(config.device), labels.to(config.device)
            # base_images_id, labels_id= base_images_id.to(config.device), labels_id.to(config.device)
            labels_emo_, _ = model(base_images)
        except ValueError:
            base_images, base_images_id, labels, labels_id = data
            base_images, labels= base_images.to(config.device), labels.to(config.device)
            # base_images_id, labels_id= base_images_id.to(config.device), labels_id.to(config.device)
            labels_emo_, _ = model(CenterCrop(size)(base_images))

        positive_pred += torch.sum(torch.argmax(labels_emo_, dim=1, keepdim=False) == labels)
        amount += len(labels)
    
    accuracy = positive_pred.cpu().numpy() / amount
    
    return accuracy
        










