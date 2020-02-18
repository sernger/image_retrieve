from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import time
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback


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

class MyModelCheckPoint(ModelCheckpoint):
    def set_model(self, model):
        self.model = model.get_layer(name='res50')

def train():
    input_shape = [160, 160, 1]
    print("input_shape:", input_shape)

    left = keras.Input(shape=input_shape, name='left')
    right = keras.Input(shape=input_shape, name='right')

    model = ResNet50_model(input_shape)

    left_out = model(left)
    right_out = model(right)

    label = keras.Input(shape=(1,), name='label')

    # custom loss layer
    loss = keras.layers.Lambda(lambda x: contrastive_loss(*x), name="loss")([left_out, right_out, label])

    siamese_model = keras.Model(inputs=[left, right, label], outputs=[left_out, right_out, loss])

    loss_output = siamese_model.get_layer("loss").output
    siamese_model.add_loss(loss_output)
    siamese_model.compile(optimizer='Adam', loss=[None, None, None])

    print(model.summary())
    print(siamese_model.summary())

    model.load_weights('saved_models\\2_resnet50_weight.031.0.00246.h5')

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = '2_resnet50_weight.{epoch:03d}.{loss:0.5f}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # save best weight
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


    #dataset[class][image_paths]
    dataset = load_data()

    log_path = './graph'
    callback = keras.callbacks.TensorBoard(log_path)
    callback.set_model(siamese_model)  
    train_names = ['train_loss', 'train_mae']
    val_names = ['val_loss', 'val_mae']

   
    compounds_per_batch = 45
    images_per_compound = 40
    alpha = 0.2

    epoch = 0
    while epoch < FLAGS.train_iter:
        batch_no = 0
        while batch_no < FLAGS.epoch_size:
            # sample compound
            image_paths, num_per_class = sample_compound(dataset, compounds_per_batch, images_per_compound)

            images = loadimgs(image_paths)

            print('Running forward pass on sampled images: ', end='')
            start_time = time.time()
            emb_array = model.predict_on_batch(images)
            print('%.3f' % (time.time()-start_time))


            print('Selecting suitable triplets for training')
            triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class, 
                        images, compounds_per_batch, alpha)
            selection_time = time.time() - start_time
            print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
                (nrof_random_negs, nrof_triplets, selection_time))

        
            # Perform training on the selected triplets
            start_time = time.time()
            [left_out, right_out, loss] = siamese_model.train_on_batch(triplets, None)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_no+1, FLAGS.epoch_size, duration, loss))
            '''
            write_log(callback, train_names, logs, batch_no)
        
            if batch_no % 10 == 0:
                X_val, Y_val = np.random.rand(32, 3), np.random.rand(32, 1)
                logs = model.train_on_batch(X_val, Y_val)
                write_log(callback, val_names, logs, batch_no//10)
            '''
            batch_no += 1

        epoch += 1
    

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def sample_compound(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class

def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    pairs = []
    for i in range(len(triplets)):
        pairs.append([triplets[i][0],triplets[i][1], 1])
        pairs.append([triplets[i][0],triplets[i][2], 0])
    return pairs, num_trips, len(pairs)

if __name__ == "__main__":
    train()
    print("")
