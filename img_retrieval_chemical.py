
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
import pandas as pd

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
    model = ResNet50_model(input_shape)
    #print(tool.Time() + "ModelAndWeight load ResNet50_model end")
    left_out = model(left)
    right_out = model(right)
    label = keras.Input(shape=(1,), name='label')
    loss = keras.layers.Lambda(lambda x: contrastive_loss(*x), name="loss")([left_out, right_out, label])
    siamese_model = keras.Model(inputs=[left, right, label], outputs=[left_out, right_out, loss])
    siamese_model.load_weights("saved_models//1_8_resnet50_model_weight.011.0.0015374334919906686.h5")
    print(tool.Time() + "ModelAndWeight load end")
    return model

if __name__ == "__main__":
    model = ModelAndWeight()
    (train_x, train_labels), (_, _)= load_dataset(320)
    train_x = np.expand_dims(train_x, axis=3) / 255.0

    train_feat = model.predict(train_x)
    del train_x
    gc.collect()
    print(train_feat.shape)

    pd_data = pd.DataFrame(data=train_feat, index=train_labels)
    pd_data.to_csv("train_feat.csv")

    pd_data1 = pd.read_csv("train_feat.csv")
    a = 0
    '''
    similar = retrieval_sim(train_feat[0:32000], train_feat)
    print(similar.shape)

    print("acc:{}%".format(np.sum(similar == np.arange(32000))/32000*100))
    #print(similar)
    '''