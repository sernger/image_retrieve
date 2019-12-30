import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


images = []
labels = []

IMAGE_SIZE = 160 # 指定图像大小
# 按指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0,0,0,0)
    # 获取图片尺寸
    h, w = image.shape
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h, w)
    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    BLACK = [255,255,255]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))

# path_name是当前工作目录，后面会由os.getcwd()获得
def read_path(path_name):
    count = 0
    for dir_item in tqdm(os.listdir(path_name), desc='dirs'): # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        count += 1
        if count > 100:
            break
        # 从当前工作目录寻找训练集图片的文件夹
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path): # 如果是文件夹，继续递归调用，去读取文件夹里的内容
            read_path(full_path)
        else: # 如果是文件了
            if dir_item.endswith('.png'):
                image = cv2.imread(full_path, 0)
                #ret, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                if image is None: # 遇到部分数据有点问题，报错'NoneType' object has no attribute 'shape'
                    pass
                else:
                    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    images.append(image)
                    labels.append(dir_item[:-4])
                    

    return np.array(images, dtype='float'), np.array(labels)
# 读取训练数据并完成标注
def load_dataset():
    images,labels = read_path("image-all")
   
   # X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0, random_state=42)
  #  X_train, y_train = data_aug_rotate(X_train, y_train)
 #   X_test, y_test = data_aug_rotate(X_test, y_test)
 #   print(X_train.shape) 
 #   return (X_train, y_train), (X_test, y_test)
    return (images, labels), (None, None)

def gasuss_noise(image, mean=0, var=0.001):
        ''' 
            添加高斯噪声
            mean : 均值 
            var : 方差
        ''' 
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, 0, 1.0)
        #cv.imshow("gasuss", out)
        return out

def rotate(image, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    angle = np.random.randint(360)
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(1.0,1.0,1.0))

    # 返回旋转后的图像
    return rotated

def image_random(image):

    image = gasuss_noise(image)
    image = rotate(image)
    return image

def data_aug_rotate(images, labels):
    aug_img=[]
    aug_lab=[]
    for i in tqdm(range(images.shape[0])):
        aug_img.append(cv2.rotate(images[i],0))
        aug_lab.append(labels[i])
        aug_img.append(cv2.rotate(images[i],1))
        aug_lab.append(labels[i])
        aug_img.append(cv2.rotate(images[i],2))
        aug_lab.append(labels[i])
    images = np.append(images, np.array(aug_img), axis=0)
    labels = np.append(labels, np.array(aug_lab), axis=0)
    return images, labels


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_dataset()
    im = X_train[2,:,:]/255
    im0 = image_random(im)
    im1 = image_random(im)
    im2 = image_random(im)
    hmerge = np.hstack((im, im0, im1, im2))
    cv2.imshow("1", hmerge)

    print(X_train[2])
    
    cv2.waitKey(10000)



