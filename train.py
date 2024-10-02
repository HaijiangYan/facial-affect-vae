# single epoch train function
import torch
from torch.cuda.amp import GradScaler, autocast  # mixed precision
import torch.nn as nn
from torchvision.transforms import CenterCrop
from tqdm import tqdm
import numpy as np


def train(model, train_loader, optimizer, scaler, config):

	model.train()  # or model.eval(), to activate or freeze the BN and dropout layers

	Loss_R, Loss_K, Loss_C_emo = 0, 0, 0

	# for data in tqdm(train_loader, ncols=100, desc="Train:"): 
	for data in train_loader: 

		if not config.aug:
			# base_images, base_images_id, labels, labels_id = data
			base_images, _, labels, _ = data
			base_images, labels= base_images.to(config.device), labels.to(config.device)
			# base_images_id, labels_id= base_images_id.to(config.device), labels_id.to(config.device)
		elif config.aug:
			base_images, _, aug_images, labels, _ = data
			base_images, aug_images, labels= base_images.to(config.device), aug_images.to(config.device), labels.to(config.device)
		else:
			raise NotImplemented

		optimizer.zero_grad()

		with autocast():

			if config.aug:
				if config.Ncrop:
					bs, ncrops, c, h, w = aug_images.shape
					aug_images = aug_images.view(-1, c, h, w)  # the first dimension becomes Ncrop x batch-size
					base_images = torch.repeat_interleave(base_images, repeats=ncrops, dim=0)
					labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
				(embeddings, _), (q_z_emo, p_z_emo), _, (images_, labels_) = model(labels_id, aug_images)  

			elif not config.aug:
				(embeddings_emo, _), (q_z_emo, p_z_emo), _, (images_, labels_emo_) = model(CenterCrop((40, 40))(base_images))
			# output of encoder (prior and posterior distribution) and decoder

			loss_recon = config.loss(reduction='none')(images_, base_images * 255).sum(-1).mean()

			loss_cla_emo = nn.CrossEntropyLoss(reduction="none")(labels_emo_, labels).sum(-1).mean()

			if model.submodel2.distribution == 'normal':
				loss_KL_emo = torch.distributions.kl.kl_divergence(q_z_emo, p_z_emo).sum(-1).mean()
			elif model.submodel2.distribution == 'vmf':
				loss_KL_emo = torch.distributions.kl.kl_divergence(q_z_emo, p_z_emo).mean()
			else:
				raise NotImplemented

			# loss = loss_recon * config.weight_recon + 0.2 * loss_KL_emo + loss_cla_emo * config.weight_class_emo
			loss = loss_recon * config.weight_recon + loss_cla_emo * config.weight_class_emo

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		Loss_R += loss_recon
		Loss_K += loss_KL_emo
		Loss_C_emo += loss_cla_emo

	return Loss_R, Loss_K, Loss_C_emo