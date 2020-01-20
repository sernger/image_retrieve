import time
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

def Time():
    return time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())



def canny_demo(image, image_out):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    if(image_out != None):
        cv.imshow("canny_output", canny_output)
        cv.imwrite(image_out, canny_output)
    return canny_output

# 轮廓外接矩形
def get_canny_demo(image, image_temp, image_analysis):
    src = cv.imread(image)
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    binary = canny_demo(src, image_temp)
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

    # 轮廓发现
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[c]);
        cv.drawContours(src, contours, c, (0, 0, 255), 2, 8) # 所有边缘
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 1, 8, 0);
        rect = cv.minAreaRect(contours[c])
        cx, cy = rect[0]
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(src, [box], 0, (0, 0, 255), 2) # 外层边框
        cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0) # 中心点

    # 显示
    cv.imshow("contours_analysis", src)
    cv.imwrite(image_analysis, src)
    cv.waitKey(0)
    cv.destroyAllWindows()


# eg: e:/image-all/273.png
# 按照轮廓截取图片
def get_canny_only_one(image, image_temp=None, image_out=None):
    src = cv.imread(image, 0)

    t = 80
    canny_output = canny_demo(src, image_temp)

    k = np.ones((3, 3), dtype=np.uint8)
    canny_output = cv.morphologyEx(canny_output, cv.MORPH_DILATE, k)

    # 轮廓发现
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[c]);
        #cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 1, 8, 0);
        if(c == 0):
            x_min = x
            y_min = y
            x_max = x+w
            y_max = y+h
        else:
            if(x < x_min):
                x_min = x
            if(x+w > x_max):
                x_max = x+w

            if(y < y_min):
                y_min = y
            if(y+h > y_max):
                y_max = y+h

    #cv.rectangle(src, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1, 8, 0);
    img = src[y_min:y_max, x_min:x_max] # 图像裁剪

    # 边界扩充
    border = 15
    constant = cv.copyMakeBorder(img,border,border,border,border,cv2.BORDER_CONSTANT,value=[255,255,255])

    # 显示
    if(image_out !=None):
        cv.imwrite(image_out, constant)
 #   cv.imshow("contours_out", constant)
  #  cv.waitKey(0)
  #cv.destroyAllWindows()
  
    return constant
    
    


#将原始图片重新处理一遍
def rechange_image(dir_source, dir_target, n=0):
    count = 0
    for dir_item in tqdm(os.listdir(dir_source), desc='dirs'):
        count += 1
        if n != 0 and count > n:
            break
        # 从当前工作目录寻找训练集图片的文件夹
        full_path = os.path.abspath(os.path.join(dir_source, dir_item))

        if os.path.isdir(full_path):
            # read_path(full_path, n)
            pass
        else:  # 如果是文件了
            if dir_item.endswith('.png'):
                sub_dir = dir_item[0:-4]
                os.mkdir(dir_target + sub_dir)
                image_out = dir_target + sub_dir + '\\' + dir_item
                get_canny_only_one(full_path, None, image_out)

    return ""

def threshold(image):
    img = cv.imread(image, 0)
    h  = img.shape[0]
    w = img.shape[1]
    ret, thresh1 = cv.threshold(img, h, w, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, h, w, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, h, w, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, h, w, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, h, w, cv.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def test_threshold():
    threshold("image-test\\2\\24-web-ta.png")

    img = get_canny_only_one("image-test\\2\\24-web-ta.png")
    ret, img = cv.threshold(img, img.shape[0], img.shape[1], cv.THRESH_TOZERO)
    filename = "image-test\\2\\24-web-ta.txt"

    row = np.array(img).shape[0]  # 获取行数n
    with open(filename, 'w') as f:  # 若filename不存在会自动创建，写之前会清空文件
        for i in range(0, row):
            f.write(str(img[i][0:]))
            f.write("\n")

    f.close()


if __name__ == "__main__":
    #get_canny_only_one("image-test\\6.png")
    rechange_image( "E:\\image-all\\", "E:\\image-new\\")
    print("")
