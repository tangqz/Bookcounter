#此文件包含训练和使用的代码,训练和使用都通过UI进行，训练时需要标出图像中含有特定特征的区域，使用时会输出图像中含有特定特征的区域
#如果可能的话，训练和使用均使用CUDA加速
#此文件的目的是使用CNN提取图像特征，然后输出图像中所有含有特定特征的区域
#具体的用途是识别一张包含一叠作业本的侧视图的图像，输出图像中每本作业本的位置
#对于使用，输入一张图像和模型，输出图像中含有特定特征的区域，用点和序号标出
#对于训练，输入训练集，创建一个UI，用鼠标标出每张图像中含有特定特征的区域，然后保存，然后训练，然后保存模型
#使用方法：python CNN.py -m use -i input.jpg -o output.jpg -w model.h5
#训练方法: python CNN.py -m train -i dataset -o model.h5
#导入必要的库，尽量使用CUDA加速
import cv2
import numpy as np
import sys
import os
import time
import math
import random
import argparse
import shutil
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras import backend as K
#以下是训练部分，输入训练集，创建一个UI，用鼠标标出每张图像中含有特定特征的区域，然后保存，然后训练，然后保存模型
#训练集的目录结构如下：
#dataset
#├───train
#│   ├───0
#│   ├───1
#│   └───2
#└───val
#    ├───0
#    ├───1
#    └───2
#其中0，1，2分别是不含有特定特征的图像，含有特定特征的图像，含有特定特征的图像
#接下来是代码：
#定义一个类，用于创建一个UI，用鼠标标出图像中含有特定特征的区域
class ImageLabeler:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()
        self.points = []
        self.window_name = image_path
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.run()
    #鼠标回调函数，用于标出图像中含有特定特征的区域
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 3, (0, 0, 255), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.pop()
            self.image = self.image_copy.copy()
            for point in self.points:
                cv2.circle(self.image, point, 3, (0, 0, 255), -1)
    #运行UI，用于标出图像中含有特定特征的区域
    def run(self):
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cv2.destroyAllWindows()
        self.save()
    #保存标注结果
    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.points, f)


#使用上面的类，对训练集中的每张图像进行标注
def label(dataset_dir):
    #创建训练集和验证集
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    #对训练集中的每张图像进行标注
    for i in range(3):
        train_class_dir = os.path.join(train_dir, str(i))
        for image_path in glob.glob(os.path.join(train_class_dir, "*.jpg")):
            output_path = image_path.replace(".jpg", ".json")
            ImageLabeler(image_path, output_path)
    #对验证集中的每张图像进行标注
    for i in range(3):
        val_class_dir = os.path.join(val_dir, str(i))
        for image_path in glob.glob(os.path.join(val_class_dir, "*.jpg")):
            output_path = image_path.replace(".jpg", ".json")
            ImageLabeler(image_path, output_path)

def train(dataset_dir, model_path):
    #使用上面的函数对训练集中的每张图像进行标注
    label(dataset_dir)
    #创建训练集和验证集
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    #创建模型
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #创建训练集和验证集的生成器
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32, class_mode='categorical')
    #训练模型
    model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=val_generator, validation_steps=50)
    #保存模型
    model.save(model_path)


#以下是使用部分，输入一张图像和模型，输出图像中含有特定特征的区域，用点和序号标出
#使用方法：python CNN.py -i image.jpg -m model.h5
def use(image_path, model_path):
    #加载模型
    model = load_model(model_path)
    #加载图像
    image = cv2.imread(image_path)
    #预处理图像
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    #预测图像
    pred = model.predict(image)
    #根据预测结果，标出图像中含有特定特征的区域
    if pred[0][0] > 0.5:
        cv2.putText(image, "class 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif pred[0][1] > 0.5:
        cv2.putText(image, "class 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif pred[0][2] > 0.5:
        cv2.putText(image, "class 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #显示结果
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#以下是主函数，根据命令行参数，执行相应的操作
def main():
    #创建命令行参数解析器
    ap = argparse.ArgumentParser()
    #添加命令行参数
    ap.add_argument("-i", "--image", help="path to the input image")
    ap.add_argument("-m", "--model", help="path to the trained model")
    ap.add_argument("-d", "--dataset", help="path to the dataset")
    #解析命令行参数
    args = vars(ap.parse_args())
    #如果命令行参数中包含image，则执行use函数
    if args["image"] is not None:
        use(args["image"], args["model"])
    #如果命令行参数中包含dataset，则执行train函数
    elif args["dataset"] is not None:
        train(args["dataset"], args["model"])