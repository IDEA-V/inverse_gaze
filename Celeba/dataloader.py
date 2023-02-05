import os, utils, torchvision
import json, PIL, time, random
import torch, math

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F 
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import h5py
import cv2

mnist_path = "./data/MNIST"
mnist_img_path = "./data/MNIST_imgs"
cifar_path = "./data/CIFAR"
cifar_img_path = "./data/CIFAR_imgs"
os.makedirs(mnist_path, exist_ok=True)
os.makedirs(mnist_img_path, exist_ok=True)

def crop(x):
    crop_size = 108
    
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    return x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.model_name = args["dataset"]["model_name"]
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if len(line) == 1:
                break
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden))

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".jpg"):
                print(f"Loading... {i}/{len(self.name_list)}", end='\r')
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                img_list.append(img)
        return img_list

    def get_processor(self):
        if self.model_name == "FaceNet":
            re_size = 112
        else:
            re_size = 64
        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        if self.mode == "gan":
            return img
        label = self.label_list[index]
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        return img, one_hot, label

    def __len__(self):
        return self.num_img

class GazeFolder(data.Dataset):

    def __init__(self, path, identities):
        self.path = path
        self.identities = identities
        self.img_list, self.labels, self.files = self.load_img()
        self.processor = transforms.ToTensor()

    def get_processor(self):
        proc = []
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((64, 64)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def load_img(self):
        id_dirs = os.listdir(self.path)
        img_list = []
        label_list = []
        file_list = []
        for dir in id_dirs:
            if int(dir[1:]) in self.identities:
                person_Dir = os.path.join(self.path, dir)
                for day in os.listdir(person_Dir):
                    # print(os.path.join(person_Dir, day))
                    mat_file = loadmat(os.path.join(person_Dir, day))
                    right_eye_images = mat_file['data'][0,0][0][0,0][1]
                    for i, img in enumerate(right_eye_images):
                        img_list.append(img)
                        label_list.append(self.identities.index(int(dir[1:])))
                        file_list.append(os.path.join(person_Dir, day) + f" {i} right")
                    left_eye_images = mat_file['data'][0,0][1][0,0][1]
                    for i, img in enumerate(left_eye_images):
                        img_list.append(img)
                        label_list.append(self.identities.index(int(dir[1:])))
                        file_list.append(os.path.join(person_Dir, day) + f" {i} left")

        return img_list, label_list, file_list
    
    def __getitem__(self, index):
        processor = self.get_processor()
        img = processor(self.img_list[index])
        return (img.repeat(3,1,1), self.labels[index], self.files[index])
    
    def __len__(self):
        return len(self.img_list)

class GazeFaceFolder(data.Dataset):

    def __init__(self, path, identities):
        self.path = path
        self.identities = identities
        self.img_list, self.labels, self.files = self.load_img()
        self.processor = transforms.ToTensor()

    def get_processor(self):
        proc = []
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((64, 64)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def load_img(self):
        files = [ name for name in os.listdir(self.path) if name.endswith('.mat')]
        img_list = []
        label_list = []
        file_list = []
        for file in files:
            filePath = os.path.join(self.path, file)
            if (int(file[1:3]) in self.identities):
                with h5py.File(filePath) as mat_file:
                    for i in range(1000):
                        img = mat_file['Data']['data'][i]
                        img = cv2.resize(img, (64, 64))
                        img_list.append(img)
                        label_list.append(0)
                        file_list.append('')

        return img_list, label_list, file_list
    
    def __getitem__(self, index):
        processor = self.get_processor()
        img = processor(self.img_list[index])
        return (img, self.labels[index], self.files[index])
    
    def __len__(self):
        return len(self.img_list)

class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden))

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('L')
                img_list.append(img)
        return img_list
    
    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "MNIST":
            re_size = 32
        else:
            re_size = 64
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        if self.mode == "gan":
            return img
        label = self.label_list[index]
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        return img, one_hot, label

    def __len__(self):
        return self.num_img

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(mnist_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(mnist_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=1)
    test_loader = DataLoader(testset, batch_size=1)
    cnt = 0

    for imgs, labels in train_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))

    for imgs, labels in test_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))

if __name__ == "__main__":
    print("ok")



    

