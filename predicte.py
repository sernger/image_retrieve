
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
def img_to_encoding(image_path, model):
    img = cv2.imread(image_path, 0)
    img = img[..., ::-1]
    img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
    images_train = np.expand_dims(img, axis=3) / 255.0
    x_train = np.array([images_train])
    embedding = model.predict_on_batch(x_train)
    return embedding


# 减小特征值数量，每4个取平均值,数量2048->512
def img_to_encoding_2(image_path, model):
    embedding = img_to_encoding(image_path, model)
    temp1 = embedding.reshape(-1, 4)
    temp2 = np.mean(temp1, axis=1)
    # temp3 = np.array([temp2])
    return temp2


def who_is_it(encoding, database):
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

    return database;

def ModelAndWeight():
    print(tool.Time() + "ModelAndWeight load begin")
    input_shape = [160, 160, 1]
    left = keras.Input(shape=input_shape, name='left')
    right = keras.Input(shape=input_shape, name='right')
    model = ResNet50_model(input_shape)
    #print(tool.Time() + "ModelAndWeight load ResNet50_model end")
    left_out = model(left)
    right_out = model(right)
    label = keras.Input(shape=(1,), name='label')
    loss = keras.layers.Lambda(lambda x: contrastive_loss(*x), name="loss")([left_out, right_out, label])
    siamese_model = keras.Model(inputs=[left, right, label], outputs=[left_out, right_out, loss])
    siamese_model.load_weights("saved_models//1_4_resnet50_model_weight.260.h5")
    print(tool.Time() + "ModelAndWeight load end")
    return model



if __name__ == "__main__":
    print("================predicte.py start ==================")
    model = ModelAndWeight()
    print("================predicte.py load_weights end ==================")


    # database = {}
    # database["6"] = img_to_encoding_2("E:/image-new/6.png", model)
    # database["7"] = img_to_encoding_2("E:/image-new/7.png", model)
    # database["11"] = img_to_encoding_2("E:/image-new/11.png", model)
    #
    # encoding = img_to_encoding_2("image-test/11.png", model)
    # who_is_it(encoding, database)
    # encoding = img_to_encoding_2("image-test/11-1.png", model)
    # who_is_it(encoding, database)
    # print("")

    encoding = img_to_encoding_2("image-test/11.png", model)
    db.who_is_it(encoding)
    print("")













