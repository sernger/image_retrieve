
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import ChemicalDataset
from model import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec
import os
import cv2

from matplotlib.pyplot import imshow

#helper function to plot image
def show_image(idxs, data):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])

    hmerge = np.hstack(data[idxs])
    cv2.imshow("1", hmerge)
    cv2.waitKey(0)

    '''
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure()
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)):
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(data[idxs[i],:,:,0])
        ax.axis('off')
    plt.show()
    '''


dataset = ChemicalDataset()
train_images = dataset.images_train
test_images = dataset.images_test
len_test = len(test_images)
len_train = len(train_images)

IMAGE_WIDTH = train_images.shape[2]
IMAGE_HEIGHT = train_images.shape[1]
img_placeholder = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='img')
net = mnist_model(img_placeholder, reuse=False)

saver = tf.train.Saver() #保存模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model.ckpt")
    
    train_feat = sess.run(net, feed_dict={img_placeholder:train_images})  

idx = np.random.randint(1, len_test)
im = test_images[idx]

#show the test image
show_image(idx, test_images)
print("This is image from id:", idx)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model.ckpt")
    search_feat = sess.run(net, feed_dict={img_placeholder:[im]})
    
#calculate the cosine similarity and sort
dist = cdist(train_feat, search_feat, 'cosine') #cdist 计算两个输入集合的距离
rank = np.argsort(dist.ravel())  #np.argsort 将x中的元素从小到大排列，提取其对应的index(索引)

#show the top n similar image from train data
n = 7
show_image(rank[:n], train_images)
print("retrieved ids:", rank[:n])

