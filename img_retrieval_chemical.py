
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from matplotlib import gridspec
import os
import keras
import keras.backend as K
from matplotlib.pyplot import imshow

from train_res import ResNet50_model
from train_res import contrastive_loss
from dataset_chemical import *
import tool
import gc 
from dataset_chemical import load_dataset

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
def retrieval_sim(search_feat, train_feat):
    dist = cdist(train_feat, search_feat, 'euclidean') #cdist 计算两个输入集合的距离
    similar = np.squeeze(np.argmin(dist,axis=0)) 
    return similar

def ModelAndWeight():
    print(tool.Time() + "ModelAndWeight load begin")
    input_shape = [160, 160, 1]
    left = keras.Input(shape=input_shape, name='left')
    right = keras.Input(shape=input_shape, name='right')
    model_ = ResNet50_model(input_shape)
    X = keras.Input(shape=input_shape)
    Y = model_(X)
    Y = keras.layers.Conv2D(128,(1,1))(Y)
    Y = keras.layers.GlobalAveragePooling2D()(Y)
    model = keras.Model(inputs=[X], outputs=[Y])

    left_out = model(left)
    right_out = model(right)

    label = keras.Input(shape=(1,), name='label')
    loss = keras.layers.Lambda(lambda x: contrastive_loss(*x), name="loss")([left_out, right_out, label])
    siamese_model = keras.Model(inputs=[left, right, label], outputs=[left_out, right_out, loss])
    siamese_model.load_weights("saved_models//1_10_resnet50_model_weight.061.0.0030255448108073325.h5")
    print(tool.Time() + "ModelAndWeight load end")
    #model.save("saved_models//predict_model.h5")
    return model

if __name__ == "__main__":
   # model = ModelAndWeight()
    model = keras.models.load_model("saved_models//1_10_new_resnet50_model_weight.001.0.004147404377086787.h5")
    (train_x, train_labels), (_, _)= load_dataset(320)
    train_x = np.expand_dims(train_x, axis=3) / 255.0
    input_shape = [160, 160, 1]
    model2 = keras.applications.ResNet50(include_top=False, input_shape=input_shape, weights=None, pooling=None)
    for i in range(len(model2.layers)):
        #if len(model2.layers[i].weights) != 0 :
        model2.layers[i].set_weights(model.layers[i].get_weights())
    model2.save_weights("saved_models\\1.9.resnet50.weight.h5")
    train_feat = model.predict(train_x)
    del train_x
    gc.collect()
    print(train_feat.shape)

    '''
    similar = retrieval_sim(train_feat[0:32000], train_feat)
    print(similar.shape)

    print("acc:{}%".format(np.sum(similar == np.arange(32000))/32000*100))
    #print(similar)
    '''