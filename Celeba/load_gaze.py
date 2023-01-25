import os
import torch
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.io import loadmat
from typing import List,Tuple


def imshow(img):
    img = img / 255
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

class GazeNormalizedDataset(Dataset):

    def __init__(self, dataSetPath:str,individuals: List[str]=None):
        self.gazePath = dataSetPath+'/'
        self.personList = glob.glob(self.gazePath + "*")
        self.data:list[Tuple[torch.tensor,str]] = []
        class_number = 0
        for personPath in self.personList:
            person_name = personPath.split("/")[-1]
            if individuals!=None and person_name not in individuals:
                continue
            person_number = class_number
            class_number+=1
            for day in glob.glob(personPath + "/*.mat"):
                mat_file = loadmat(day)
                right_eye_images = mat_file['data'][0,0][0][0,0][1]
                eye_images_list = list(right_eye_images)
                sample_list_with_label = [[eye_image,person_number] for eye_image in eye_images_list]
                self.data.extend(sample_list_with_label)
        self.class_count = class_number

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        #repeat for 3 channel
        img = np.repeat(sample[0].astype('float32').reshape(1,36,60),3,axis=0)

        return [img,sample[1]]