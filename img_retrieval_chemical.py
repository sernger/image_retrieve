
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

K.set_image_data_format('channels_last')
K.set_floatx('float32')

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
def retrieval_sim(search_feat, train_feat):
    dist = cdist(train_feat, search_feat, 'euclidean') #cdist 计算两个输入集合的距离
    similar = np.squeeze(np.argmin(dist,axis=0)) 
    return similar

def ModelAndWeight():
    print(tool.Time() + "ModelAndWeight load begin")
    input_shape = [160, 160, 1]
    model = ResNet50_model(input_shape)
    model.load_weights("saved_models//1_10_new_resnet50_model_weight.001.0.06478.h5")
    print(tool.Time() + "ModelAndWeight load end")
    return model

if __name__ == "__main__":
    model = ModelAndWeight()
   
    (train_x, train_labels), (_, _)= load_dataset(0)
    train_x = np.expand_dims(train_x, axis=3) / 255.0

    train_feat = model.predict(train_x)
    del train_x
    gc.collect()
    print(train_feat.shape)

    np.savez("feat_label", train_feat, train_labels)
    #train_feat, train_labels = np.load("feat_label.npz")
    search_feat = train_feat[0:320]
    similar = retrieval_sim(search_feat, train_feat)
    print(similar.shape)

    print("acc:{}%".format(np.sum(similar == np.arange(search_feat.shape[0]))/search_feat.shape[0]*100))
    print(similar)
    