from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from tool import resize_image, IMAGE_SIZE
from model_compound import ModelAndWeight, retrieval_sim, retrieval_dissim, img_to_encoding, IMAGE_SIZE


class ChemicalDataset():
	def __init__(self, model, dir, n=0):
		print("===Loading Chemical Dataset===")
		self.labels_train = read_labels_all(dir)
		self.similarIndex = 0
		self.dissimilarIndex = 0
		self.groupSize = 10
		self.model = model


	class ImageClass():
        #"Stores the paths to images for a given class"
		def __init__(self, name, image_paths):
			self.name = name
			self.image_paths = image_paths
	
		def __str__(self):
			return self.name + ', ' + str(len(self.image_paths)) + ' images'
	
		def __len__(self):
			return len(self.image_paths)

	def get_dataset(self, path):
		dataset = []
		path_exp = os.path.expanduser(path)
		classes = [path for path in os.listdir(path_exp) \
						if os.path.isdir(os.path.join(path_exp, path))]
		classes.sort()
		nrof_classes = len(classes)
		for i in range(nrof_classes):
			class_name = classes[i]
			facedir = os.path.join(path_exp, class_name)
			image_paths = self.get_image_paths(facedir)
			dataset.append(self.ImageClass(class_name, image_paths))
	
		return dataset

	def get_image_paths(facedir):
		image_paths = []
		if os.path.isdir(facedir):
			images = os.listdir(facedir)
			image_paths = [os.path.join(facedir,img) for img in images]
		return image_paths

	def getLen(self):
		return self.images_train.shape[0]

	def _get_siamese_similar_pair(self):
		# 采样10组相似pair时，先获取子文件下的所有图片（即被标记的文件），数量不够时用imgaug 生成
		# 先从被标记的文件中获取
		label = self.labels_train[self.similarIndex]  # 'e:\image\6'
		l = []
		r = []
		result = np.ones(self.groupSize)
		images = read_imgs(label)
		if(len(images) < self.groupSize):
			trans = image_randoms(images[0], self.groupSize - len(images))
			images.extend(trans)

		for i in range(0, len(images)-1, 2):
			l.append(images[i])
			r.append(images[i+1])
		self.similarIndex += 1
		return l, r, result

	def _get_siamese_dissimilar_pair(self, batchSize):
		#  采样10组不相似pair时，用现有model挑选batch中表现最差的10个图片
		label = self.labels_train[self.dissimilarIndex]  # 'e:\image\6'
		l = []
		r = []
		result = np.zeros(self.groupSize)
		images = read_imgs(label)
		if(len(images) < self.groupSize):
			trans = image_randoms(images[0], self.groupSize - len(images))
			images.extend(trans)

		l = images

		search_feat = []
		for i in range(0, len(images)-1, 1):
			search_feat[i] = img_to_encoding(images[i], self.model) #model = ModelAndWeight()

		train_feat = []
		batch_label_start = self.dissimilarIndex * batch_size;
		for i in range(batch_label_start, batch_size, 1):
			train_feat[i] = img_to_encoding(read_imgs(self.labels_train[i])[0], self.model)

		dissimilar, dist = retrieval_dissim(search_feat, train_feat)
		r = label[dissimilar]

		self.dissimilarIndex += 1
		return l, r, result
	
	def _get_siamese_pair(self, batchSize):
		# if np.random.random() < 0.5:
		# 	return self._get_siamese_similar_pair()
		# else:
			return self._get_siamese_dissimilar_pair(batchSize)

	def get_siamese_batch(self, n):
		left, right, labels = [], [], []
		for _ in range(n):
			l, r, x = self._get_siamese_pair(n)
			left.append(l)
			right.append(r)
			labels.append(x)
		return [np.concatenate(left), np.concatenate(right), np.expand_dims(labels, axis=1)]




# 获取所有文件目录，并排序
# 返回绝对路径：'e:\\image-new\\6'
# read_labels_all("e:\\image-new\\")
def read_labels_all(path_name):
	labels = glob(path_name + "*")
	labels.sort(key=lambda x: int(x[len(path_name):]))
	dirs =  np.array(labels)
	return dirs


# 获取某个子文件下（'e:\\image-new\\6'）所有文件,并归一化
def read_imgs(label_absolute_dir):
	imgs = []
	files = glob(label_absolute_dir + "\\*")
	for file in files:
		image = cv2.imread(file, 0) # 灰度图片shape=(160,160)
		image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
		image = np.expand_dims(image, axis=3) / 255.0  # liwei add
		imgs.append(np.array(image, dtype='float32'))
	return imgs


def gasuss_noise(image, mean=0, var=0.0001):
        '''
            添加高斯噪声
            mean : 均值
            var : 方差
        '''
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, 0, 1.0)
        #cv.imshow("gasuss", out)
        return out

def rotate(image, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    angle = np.random.randint(0,20) - 10
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(1.0,1.0,1.0))

    # 返回旋转后的图像
    return rotated

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
     #   shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

# def image_random(image):
#     image = gasuss_noise(image)
#     #image = rotate(image)
#     gener = datagen.flow(image, batch_size=1)
#     return gener.next()

def image_randoms(image, similar_size):
    image = np.expand_dims(image, axis=0)
    image = gasuss_noise(image)
    gener = datagen.flow(image, batch_size=1)
    images = []
    for i in range(similar_size):
        images.append(gener.next())
    return images


if __name__ == "__main__":
    model  = ModelAndWeight()
    dataset = ChemicalDataset(model, "d:\\prj\\image-new\\")
    batch_size = 16
    ls, rs, xs = dataset.get_siamese_batch(batch_size)
    print("")
