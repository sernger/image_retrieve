from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator


class Dataset(object):
	images_train = np.array([])
	images_test = np.array([])
	labels_train = np.array([])
	labels_test = np.array([])
	unique_train_label = np.array([])
	map_train_label_indices = dict()

class ChemicalDataset(Dataset):
	def __init__(self, dir, n=0):
		print("===Loading Chemical Dataset===")
		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = load_dataset(dir, n)
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
		#label = np.random.choice(self.unique_train_label)
		label = self.unique_train_label[self.dissimilarIndex%self.unique_train_label.shape[0]]  #self.unique_train_label.shape=(32000,)
		self.similarIndex += 1
		l = np.random.choice(self.map_train_label_indices[label])
		r = np.random.choice(self.map_train_label_indices[label])
		l = image_random(self.images_train[l:l+1])
		r = image_random(self.images_train[r:r+1])
		return l, r, 1

	def _get_siamese_dissimilar_pair(self):
		label_l = self.unique_train_label[self.dissimilarIndex%self.unique_train_label.shape[0]]
		self.dissimilarIndex += 1
	#	label_r = np.random.choice(self.unique_train_label[np.where(self.unique_train_label!=label_l)], 1, replace=False)
		label_r = np.random.choice(self.unique_train_label[np.where(self.unique_train_label!=label_l)])
		l = np.random.choice(self.map_train_label_indices[label_l])
		r = np.random.choice(self.map_train_label_indices[label_r])

		l = image_random(self.images_train[l:l+1])
		r = image_random(self.images_train[r:r+1])
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
		return [np.concatenate(left), np.concatenate(right), np.expand_dims(labels, axis=1)]

def read_imgs(path_name, label):
    full_path = os.path.abspath(os.path.join(path_name, label+'.png'))
    image = cv2.imread(full_path, 0)
    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
    return np.array(image, dtype='float32')

def load_dataset(dir, n=0):
    # images.shape=(32000, 160, 160)  labels.shape=(32000,)
    images,labels = read_path(dir, n)
    return (images, labels), (None, None)


images = []
labels = []
IMAGE_SIZE = 160 # 指定图像大小
# 按指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0,0,0,0)
    # 获取图片尺寸
    h, w = image.shape
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = int(max(h, w)*1.1)
    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    WITHE = [255,255,255]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = WITHE)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))

# path_name是当前工作目录，后面会由os.getcwd()获得
def read_path(path_name , n=0):
    count = 0
    for dir_item in tqdm(os.listdir(path_name), desc='dirs'):
        count += 1
        if n != 0 and count > n:
            break
        # 从当前工作目录寻找训练集图片的文件夹
        #full_path = os.path.abspath(os.path.join(path_name, dir_item[0:-4] + "//" + dir_item))
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            #read_path(full_path, n)
            pass
        else: # 如果是文件了
            if dir_item.endswith('.png'):
                image = cv2.imread(full_path, 0)
                #ret, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                if image is None:
                    pass
                else:
                    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    images.append(image)
                    labels.append(dir_item[:-4])
    return np.array(images, dtype='float32'), np.array(labels)

def read_labels(path_name):
    labels = []
    for dir_item in tqdm(os.listdir(path_name), desc='dirs'):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        labels.append(dir_item[:-4])
    return np.array(labels)

def read_imgs(path_name, label):
    full_path = os.path.abspath(os.path.join(path_name, label+'.png'))
    image = cv2.imread(full_path, 0)
    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
    return np.array(image, dtype='float32')

def load_dataset(dir, n=0):
    # images.shape=(32000, 160, 160)  labels.shape=(32000,)
    images,labels = read_path(dir, n)
    return (images, labels), (None, None)

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

def image_random(image):
    image = gasuss_noise(image)
    #image = rotate(image)
    gener = datagen.flow(image, batch_size=1)
    return gener.next()


if __name__ == "__main__":
	# Test if it can load the dataset properly or not. use the train.py to run the training
	a = ChemicalDataset(100)
	batch_size = 4
	ls, rs, xs = a.get_siamese_batch(batch_size)
	f, axarr = plt.subplots(batch_size, 2, figsize=(10, 10))
	for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
		print("Row", idx, "Label:", "similar" if x else "dissimilar")
		print("max:", np.squeeze(l).max())
		axarr[idx, 0].imshow(np.squeeze(l),cmap='gray', vmin=0, vmax=1.0)
		axarr[idx, 1].imshow(np.squeeze(r),cmap='gray', vmin=0, vmax=1.0)
	plt.show()
'''
	hmerge1 = np.vstack(ls)
	hmerge2 = np.vstack(rs)
	hmerge = np.hstack((hmerge1, hmerge2))
	cv2.imshow("1", hmerge)
	cv2.waitKey(0)
	'''