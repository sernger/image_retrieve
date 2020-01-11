from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint



from dataset import MNISTDataset, ChemicalDataset
from functools import partial
#from model import *
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('train_data_size', 32000, 'train data size.')
#flags.DEFINE_integer('val_data_size', 320, 'val data size.')
flags.DEFINE_integer('train_iter', 500, 'Total training iter')
#flags.DEFINE_integer('step', 50, 'Save after ... iteration')
#flags.DEFINE_string('model', 'mnist', 'model to run')

K.set_image_data_format('channels_last')
K.set_floatx('float32')
def ResNet50_model(input_shape):
    res_model = keras.applications.ResNet50(include_top=False, input_shape=input_shape, weights=None, pooling=None) #pooling='max')
    X = keras.Input(shape=input_shape)
    Y = res_model(X)
    Y = keras.layers.Conv2D(128,(1,1))(Y)
    Y = keras.layers.GlobalAveragePooling2D()(Y)
    model = keras.Model(inputs=[X], outputs=[Y], name="res50")
   # plot_model(res_model,to_file="Resnet50.png",show_shapes=True)
    return model

def contrastive_loss(l, r, y):
    margin = 0.5
    distance = K.sqrt(K.sum(K.pow(l - r, 2), 1, keepdims=True))
    similarity = y * K.square(distance)                                           # keep the similar label (1) close to each other
    dissimilarity = (1 - y) * K.square(K.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
    return K.mean(dissimilarity + similarity)

def next_generator():
    data = ChemicalDataset(FLAGS.train_data_size)
    while True:
        yield (data.get_siamese_batch(FLAGS.batch_size), None)

class MyModelCheckPoint(ModelCheckpoint):
    def set_model(self, model):
        self.model = model.get_layer(name='res50')

if __name__ == "__main__":

    input_shape = [160,160,1]
    print("input_shape:", input_shape)

    left = keras.Input(shape=input_shape, name='left')
    right = keras.Input(shape=input_shape, name='right')

    model = ResNet50_model(input_shape)
    
    left_out = model(left)
    right_out = model(right)

    label = keras.Input(shape=(1,), name='label')

    #custom loss layer
    loss=keras.layers.Lambda(lambda x:contrastive_loss(*x), name="loss")([left_out, right_out, label])

    siamese_model = keras.Model(inputs=[left, right, label], outputs=[left_out, right_out, loss])

    loss_output= siamese_model.get_layer("loss").output
    siamese_model.add_loss(loss_output)
    siamese_model.compile(optimizer='Adam', loss=[None,None,None])

    print(model.summary())
    print(siamese_model.summary())

    model.load_weights('saved_models\\1_resnet50_weight.031.0.00246.h5')
    
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = '2_resnet50_weight.{epoch:03d}.{loss:0.5f}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    
    #save best weight 
    checkpoint = MyModelCheckPoint(filepath=filepath,
                                monitor='loss',
                                verbose=1,
                                period=1,
                                save_weights_only=True,
                                save_best_only=True)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./graph', 
                                             histogram_freq=0, 
                                             write_graph=True, 
                                             write_images=True)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss')

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    callbacks = [checkpoint, tbCallBack, reduce_lr, early_stopping]

    data_gen = next_generator()

    siamese_model.fit_generator(data_gen, 
                                steps_per_epoch=FLAGS.train_data_size/FLAGS.batch_size, 
                                epochs=FLAGS.train_iter, 
                                verbose=1, 
                                #workers=4, 
                              #  validation_data = val_data,
                              # use_multiprocessing=True,
                                callbacks=callbacks)

   # model.save(os.path.join(save_dir, "resnet50_model.h5"))
