
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import ChemicalDataset
from model import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec
import os
import cv2
import keras
import keras.backend as K
from matplotlib.pyplot import imshow

from train_res import ResNet50_model
from train_res import contrastive_loss
from dataset_chemical import *
import tool
import db
from img_retrieval_chemical import ModelAndWeight

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
def img_to_encoding(image_path, model):
    img = cv2.imread(image_path, 0)
    img = img[..., ::-1]
    img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
    images_train = np.expand_dims(img, axis=3) / 255.0
    x_train = np.array([images_train])
    embedding = model.predict_on_batch(x_train)
    return embedding #shape=(1,128)


# 减小特征值数量，每4个取平均值,数量2048->512
def img_to_encoding_2(image_path, model):
    embedding = img_to_encoding(image_path, model)
    temp1 = embedding.reshape(-1, 4)
    temp2 = np.mean(temp1, axis=1) # shape=(32,)
    temp3 = np.array([temp2]) # shape=(1,32)  保持与img_to_encoding返回维度相同
    return temp2


def who_is_it_0(encoding, database):
    margin = 0.5
    # distance = K.sqrt(K.sum(K.pow(l - r, 2), 1, keepdims=True))
    # similarity = y * K.square(distance)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > margin:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ".png, the distance is " + str(min_dist))

    return min_dist, identity

def who_is_it(search_feat, train_feat):
    dist = cdist(train_feat, search_feat, 'cosine') #cdist 计算两个输入集合的距离
    rank = np.argsort(dist.ravel())  #np.argsort 将x中的元素从小到大排列，提取其对应的index(索引)
    print(rank)

def img_to_encoding_from_dir(path_name, model, n=0):
    database = {}
    count = 0
    for dir_item in tqdm(os.listdir(path_name), desc='dirs'):  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        count += 1
        if n != 0 and count > n:
            break
        # 从当前工作目录寻找训练集图片的文件夹
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用，去读取文件夹里的内容
            read_path(full_path, n)
        else:  # 如果是文件了
            if dir_item.endswith('.png'):
                fileName = dir_item[:-4]
                database[fileName] = img_to_encoding(full_path, model)

    return database


if __name__ == "__main__":
    print("================predicte.py start ==================")
    model = ModelAndWeight()
    print("================predicte.py load_weights end ==================")


    # database = {}
    # database["6"] = img_to_encoding("E:/image-all/6.png", model)
    # database["15"] = img_to_encoding("E:/image-all/15.png", model)
    # database["273"] = img_to_encoding("E:/image-all/273.png", model)
    #

    encoding = img_to_encoding("image-test/6.png", model)
    db.who_is_it(encoding[0], '6.png')
    encoding = img_to_encoding("image-test/15.png", model)
    db.who_is_it(encoding[0], '15.png')
    encoding = img_to_encoding("image-test/273.png", model)
    db.who_is_it(encoding[0], '273.png')

    encoding = img_to_encoding("image-test/6-auto-cut.png", model)
    #who_is_it_0(encoding, database)
    db.who_is_it(encoding[0], '6-auto-cut.png')
    encoding = img_to_encoding("image-test/15-auto-cut.png", model)
    #who_is_it_0(encoding, database)
    db.who_is_it(encoding[0], '15-auto-cut.png')
    encoding = img_to_encoding("image-test/273-auto-cut.png", model)
    #who_is_it_0(encoding, database)
    db.who_is_it(encoding[0], '273-auto-cut.png')
    print("")




    print("")













