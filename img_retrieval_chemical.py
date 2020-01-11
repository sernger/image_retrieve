
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
    model.load_weights("saved_models//2_resnet50_weight.049.0.00239.h5")
    print(tool.Time() + "ModelAndWeight load end")
    return model

def img_to_encoding(image_path, model):
    img = tool.get_canny_only_one(image_path)
   # img = cv2.imread(image_path, 0)
    #img = img[..., ::-1]
    cv2.imshow('countour', img)
    cv2.waitKey(10000)
    img = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
    cv2.imshow('resize_image', img)
    cv2.waitKey(10000)
    images_train = np.expand_dims(img, axis=3) / 255.0
    x_train = np.array([images_train])
    embedding = model.predict_on_batch(x_train)
    return embedding #shape=(1,128)

if __name__ == "__main__":
    model = ModelAndWeight()
   
    '''
    (train_x, train_labels), (_, _)= load_dataset(32000)
    train_x = np.expand_dims(train_x, axis=3) / 255.0
    train_feat = model.predict(train_x)
    np.savez("saved_models\\feat_label", f=train_feat, l=train_labels)
    '''
    
    np_data = np.load("saved_models\\feat_label.npz")
    train_feat = np_data['f']
    train_labels = np_data['l']
    print(train_feat.shape)

    search_feat = img_to_encoding("image-test\\49.png" ,model)
    similar = retrieval_sim(search_feat, train_feat)
    print(similar.shape)

    #print("acc:{}%".format(np.sum(similar == np.arange(search_feat.shape[0]))/search_feat.shape[0]*100))
    print(train_labels[similar])
    