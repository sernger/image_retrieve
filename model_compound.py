from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import keras
import keras.backend as K
import tool
from scipy.spatial.distance import cdist
import cv2
from tool import resize_image

IMAGE_SIZE = 160 #指定图像大小
K.set_image_data_format('channels_last')
K.set_floatx('float32')

def ModelAndWeight():
    print(tool.Time() + "ModelAndWeight load begin")
    input_shape = [IMAGE_SIZE, IMAGE_SIZE, 1]
    model = ResNet50_model(input_shape)
    model.load_weights("saved_models//3_regularizers_resnet50_weight.027.0.00243.h5")
    print(tool.Time() + "ModelAndWeight load end")
    return model

def retrieval_sim(search_feat, train_feat, n=10):
    dist = cdist(train_feat, search_feat, 'euclidean') #cdist 计算两个输入集合的距离
   #similar = np.squeeze(np.argmin(dist,axis=0))
    similar = np.argsort(dist, axis=0)[:n]
    dist = np.sort(dist, axis=0)[:n]
    return similar, dist

def retrieval_dissim(search_feat, train_feat, n=10):
    dist = cdist(train_feat, search_feat, 'euclidean') #cdist 计算两个输入集合的距离
    similar = np.argsort(dist, axis=0)[n:-1]
    dist = np.sort(dist, axis=0)[n:-1]
    return similar, dist

def img_to_encoding(image_path, model):
    img = tool.get_canny_only_one(image_path)
   # img = cv2.imread(image_path, 0)
    #img = img[..., ::-1]
    cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
    cv2.imshow('resize_image', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    images_train = np.expand_dims(img, axis=3) / 255.0
    x_train = np.array([images_train])
    embedding = model.predict_on_batch(x_train)
    return embedding #shape=(1,128)

def ResNet50_model(input_shape):
    res_model = keras.applications.ResNet50(include_top=False, input_shape=input_shape, weights=None, pooling=None) #pooling='max')
    X = keras.Input(shape=input_shape)
    Y = res_model(X)
    Y = keras.layers.Conv2D(128,(1,1))(Y)
    Y = keras.layers.GlobalAveragePooling2D()(Y)
    model = keras.Model(inputs=[X], outputs=[Y], name="res50")
   # plot_model(res_model,to_file="Resnet50.png",show_shapes=True)
    return model

