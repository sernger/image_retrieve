from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K


from dataset import MNISTDataset, ChemicalDataset
from functools import partial
#from model import *
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_string('model', 'mnist', 'model to run')

K.set_image_data_format('channels_last')
def ResNet50_model(input_shape):
    x = keras.Input(shape=input_shape)
    y = keras.applications.ResNet50(include_top=False, input_tensor=x, weights=None, pooling='max')(x)
    #y = keras.layers.Flatten()(y)
    return keras.Model(inputs=x, outputs=y)

def contrastive_loss(l, r, y):
    margin = 0.5
   # l, r, y = y_pred
    distance = K.sqrt(K.sum(K.pow(l - r, 2), 1, keepdims=True))
    similarity = y * K.square(distance)                                           # keep the similar label (1) close to each other
    dissimilarity = (1 - y) * K.square(K.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
    return K.mean(dissimilarity + similarity) / 2

def next_generator():
    data = ChemicalDataset()
    batch_size = 32
    while True:
        yield ((data.get_siamese_batch(batch_size)), None)

if __name__ == "__main__":


    #dataset = ChemicalDataset()
    input_shape = [160,160,1]
    #list(dataset.images_train.shape[1:])
    print("input_shape:", input_shape)
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    left = keras.Input(shape=input_shape, name='left')
    right = keras.Input(shape=input_shape, name='right')
    #left = tf.placeholder(tf.float32, placeholder_shape, name='left')
    #right = tf.placeholder(tf.float32, placeholder_shape, name='right')
    model = ResNet50_model(input_shape)
    left_out = model(left)
    right_out = model(right)


    label = keras.Input(shape=(1,), name='label')
   # label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
    #label_float = keras.backend.cast_to_floatx(label)

    #custom loss layer
    loss=keras.layers.Lambda(lambda x:contrastive_loss(*x), name="loss")([left_out, right_out, label])

    siamese_model = keras.Model(inputs=[left, right, label], outputs=[left_out, right_out, loss])

    loss_layer = siamese_model.get_layer("loss").output
    model.add_loss(loss_layer)
    siamese_model.compile(optimizer='Adam', loss=[None,None,None])

    siamese_model.fit_generator(next_generator, steps_per_epoch=10, epochs=200)
