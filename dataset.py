from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from dataset_chemical import load_dataset, image_random


class Dataset(object):
	images_train = np.array([])
	images_test = np.array([])
	labels_train = np.array([])
	labels_test = np.array([])
	unique_train_label = np.array([])
	map_train_label_indices = dict()

	def _get_siamese_similar_pair(self):
		label = np.random.choice(self.unique_train_label)
		if(self.map_train_label_indices[label].shape[0]<2):
		 print('errirroro'+label)
		l, r = 	np.random.choice(self.map_train_label_indices[label], 2, replace=False)
		return l, r, 1

	def _get_siamese_dissimilar_pair(self):
		label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
		l = np.random.choice(self.map_train_label_indices[label_l])
		r = np.random.choice(self.map_train_label_indices[label_r])
		return l, r, 0

	def _get_siamese_pair(self):
		if np.random.random() < 0.5:
			return self._get_siamese_similar_pair()
		else:
			return self._get_siamese_dissimilar_pair()

	def get_siamese_batch(self, n):
		idxs_left, idxs_right, labels = [], [], []
		for _ in range(n):
			l, r, x = self._get_siamese_pair()
			idxs_left.append(l)
			idxs_right.append(r)
			labels.append(x)
		return self.images_train[idxs_left,:], self.images_train[idxs_right, :], np.expand_dims(labels, axis=1)

class MNISTDataset(Dataset):
	def __init__(self):
		print("===Loading MNIST Dataset===")
		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = mnist.load_data()
		self.images_train = np.expand_dims(self.images_train, axis=3) / 255.0
		self.images_test = np.expand_dims(self.images_test, axis=3) / 255.0
		self.labels_train = np.expand_dims(self.labels_train, axis=1)
		self.unique_train_label = np.unique(self.labels_train)
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
		# print("Map label indices:", self.map_train_label_indices)

class ChemicalDataset(Dataset):
	def __init__(self, n=0):
		print("===Loading Chemical Dataset===")
		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = load_dataset(n)
		self.images_train = np.expand_dims(self.images_train, axis=3) / 255.0
#		self.images_test = np.expand_dims(self.images_test, axis=3) / 255.0
		self.labels_train = np.expand_dims(self.labels_train, axis=1)
		self.unique_train_label = np.unique(self.labels_train)
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
#		print("Images test  :", self.images_test.shape)
#		print("Labels test  :", self.labels_test.shape)
		self.similarIndex = 0
		self.dissimilarIndex = 0
		#print("Unique label :", self.unique_train_label)
		#print("Map label indices:", self.map_train_label_indices)

	def getLen(self):
		return self.images_train.shape[0]

	def _get_siamese_similar_pair(self):
		#label = np.random.choice(self.unique_train_label)
		label = self.unique_train_label[self.dissimilarIndex%self.unique_train_label.shape[0]]
		self.similarIndex += 1
		l = np.random.choice(self.map_train_label_indices[label])
		r = np.random.choice(self.map_train_label_indices[label])
		l = image_random(self.images_train[l])
		r = image_random(self.images_train[r])
		return l, r, 1

	def _get_siamese_dissimilar_pair(self):
		label_l = self.unique_train_label[self.dissimilarIndex%self.unique_train_label.shape[0]]
		self.dissimilarIndex += 1
	#	label_r = np.random.choice(self.unique_train_label[np.where(self.unique_train_label!=label_l)], 1, replace=False)
		label_r = np.random.choice(self.unique_train_label[np.where(self.unique_train_label!=label_l)])
		l = np.random.choice(self.map_train_label_indices[label_l])
		r = np.random.choice(self.map_train_label_indices[label_r])

		l = image_random(self.images_train[l])
		r = image_random(self.images_train[r])
		return l, r, 0
	
	def _get_siamese_pair(self):
		if np.random.random() < 0.5:
			return self._get_siamese_similar_pair()
		else:
			return self._get_siamese_dissimilar_pair()

	def get_siamese_batch(self, n):
		left, right, labels = [], [], []
		for _ in range(n):
			l, r, x = self._get_siamese_pair()
			left.append(l)
			right.append(r)
			labels.append(x)
		return [np.expand_dims(left, axis=3), np.expand_dims(right, axis=3), np.expand_dims(labels, axis=1)]
		
if __name__ == "__main__":
	# Test if it can load the dataset properly or not. use the train.py to run the training
	a = ChemicalDataset()
	batch_size = 4
	ls, rs, xs = a.get_siamese_batch(batch_size)
	f, axarr = plt.subplots(batch_size, 2, figsize=(10, 10))
	for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
		print("Row", idx, "Label:", "similar" if x else "dissimilar")
		print("max:", np.squeeze(l, axis=2).max())
		axarr[idx, 0].imshow(np.squeeze(l, axis=2),cmap='gray', vmin=0, vmax=1.0)
		axarr[idx, 1].imshow(np.squeeze(r, axis=2),cmap='gray', vmin=0, vmax=1.0)
	plt.show()

	hmerge1 = np.vstack(ls)
	hmerge2 = np.vstack(rs)
	hmerge = np.hstack((hmerge1, hmerge2))
	cv2.imshow("1", hmerge)
	cv2.waitKey(0)