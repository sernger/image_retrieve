import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

images = []
labels = []
IMAGE_SIZE = 160 # 指定图像大小
# 按指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0,0,0,0)
    # 获取图片尺寸
    h, w = image.shape
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = int(max(h, w)*1.3)
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
    WITHE = [255,255,255]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = WITHE)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))

# path_name是当前工作目录，后面会由os.getcwd()获得
def read_path(path_name , n=0):
    count = 0
    for dir_item in tqdm(os.listdir(path_name), desc='dirs'): 
        count += 1
        if n != 0 and count > n:
            break
        # 从当前工作目录寻找训练集图片的文件夹
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path): 
            #read_path(full_path, n)
            pass
        else: # 如果是文件了
            if dir_item.endswith('.png'):
                image = cv2.imread(full_path, 0)
                #ret, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                if image is None: 
                    pass
                else:
                    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    images.append(image)
                    labels.append(dir_item[:-4])
    return np.array(images, dtype='float32'), np.array(labels)

def load_dataset(n=0):
    images,labels = read_path("image-all//", n)
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
    angle = np.random.randint(0,20) - 10
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(1.0,1.0,1.0))

    # 返回旋转后的图像
    return rotated

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
     #   shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

def image_random(image):
    #image = gasuss_noise(image)
    #image = rotate(image)
    gener = datagen.flow(image, batch_size=1)
    return gener.next()

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_dataset(100)
    X_train = np.expand_dims(X_train, axis=3)/255.0
    i = np.random.randint(0,100)
    im = X_train[i:i+1]
    
    im0 = np.squeeze(image_random(im))
    im1 = np.squeeze(image_random(im))
    im2 = np.squeeze(image_random(im))
    im = np.squeeze(im)
    hmerge = np.hstack((im, im0, im1, im2))
    cv2.imshow("1", hmerge)

    print(im2.shape)
    
    cv2.waitKey(0)



