import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = imgs[0].shape[0], imgs[0].shape[1]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	for i in range(len(imgs)):
		imgs[i] = imgs[i][0:H, 0:W, :]

	return imgs

#image to image
class PairLoader(Dataset):
	def __init__(self, data_dir, mode, size=[100,100], edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = data_dir
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir)))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)
		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]

		source_img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, img_name if self.mode == "test" or self.mode == "valid" else f"{img_name.split('.')[0].split('_')[0]}.{img_name.split('.')[-1]}")) * 2 - 1
		
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


