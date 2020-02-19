import numpy as np
from glob import glob
import cv2
from tool import resize_image


IMAGE_SIZE = 160 # 指定图像大小
# 获取所有文件目录，并排序
# 返回绝对路径：'e:\\image-new\\6'
# load_data("e:\\image-new\\")
# return dataset[class][image_paths]
def load_data(path_name):
    dataset = []
    labels = glob(path_name + "*")
    labels.sort(key=lambda x: int(x[len(path_name):]))
    for label in labels:
        files = glob(label + "\\*")
        dataset.append(files)
    return dataset

# loadimgs(image_paths)
def loadimgs(image_paths):
    imgs = []
    for file in image_paths:
        image = cv2.imread(file, 0)  # 灰度图片shape=(160,160)
        image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
        image = np.expand_dims(image, axis=3) / 255.0  # liwei add
        imgs.append(np.array(image, dtype='float32'))
    return imgs

# 获取某个子文件下（'e:\\image-new\\6'）所有文件,并归一化
def loadimgs2(file_dir):
	imgs = []
	files = glob(file_dir + "\\*")
	for file in files:
		image = cv2.imread(file, 0) # 灰度图片shape=(160,160)
		image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
		image = np.expand_dims(image, axis=3) / 255.0  # liwei add
		imgs.append(np.array(image, dtype='float32'))
	return imgs


