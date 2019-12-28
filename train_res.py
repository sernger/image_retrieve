from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import MNISTDataset, Cifar10Dataset
#from model import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_string('model', 'mnist', 'model to run')


def ResNet50_model(input_shape):
    x = tf.keras.Input(shape=input_shape)
    y = tf.keras.applications.ResNet50(include_top=False, input=x, pooling='max')
    y = tf.contrib.layers.flatten()(y)
    return tf.keras.Model(inputs=x, outputs=y)

def contrastive_loss(model1, model2, y, margin):
    	with tf.name_scope("contrastive-loss"):
		distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
		similarity = y * tf.square(distance)                                           # keep the similar label (1) close to each other
		dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
		return tf.reduce_mean(dissimilarity + similarity) / 2


if __name__ == "__main__":
    dataset = Cifar10Dataset()
    
    placeholder_shape = [None] + list(dataset.images_train.shape[1:])
    print("cifar10 placeholder_shape", placeholder_shape)
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    next_batch = dataset.get_siamese_batch
    left = tf.placeholder(tf.float32, placeholder_shape, name='left')
    right = tf.placeholder(tf.float32, placeholder_shape, name='right')
    model = ResNet50_model(placeholder_shape)
    left_out = model(left)
    right_out = model(right)


 

    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
        label_float = tf.to_float(label)
    
    siamese_model = tf.keras.Model(inputs=[left, right])
    #left_model = tf.keras.Model(inputs=left, outputs=left_out)
    #right_model = tf.keras.Model(input=right, outputs=right_out)
    
    margin = 0.5
    loss = contrastive_loss(left_out, right_out, label_float, margin)

    left_model.compile(optimizer='Adam', loss=loss)
