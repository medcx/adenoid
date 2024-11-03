import glob
import os
from builtins import len

from torchvision.transforms import transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import json


class Patient_dataset(Dataset):
    def __init__(self, patient, target, transform=None):
        self.patient_all = patient
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.patient_all)

    def __getitem__(self, idx):
        patient = self.patient_all[idx]
        patient = patient.replace('\\', '/')
        patient_id = patient.split('/')[-1]
        patient_type = patient.split('/')[-3].split('_')[0]
        p_0 = f'../data/{patient_type}_0/train/{patient_id}/*'
        p_1 = f'../data/{patient_type}_1/train/{patient_id}/*'
        p_2 = f'../data/{patient_type}_2/train/{patient_id}/*'
        data_0 = glob.glob(p_0)
        random.shuffle(data_0)
        if len(data_0) == 0:
            d_0 = np.zeros((512, 512, 3))
        else:
            d_0 = np.array(Image.open(data_0[0]))

        data_1 = glob.glob(p_1)
        random.shuffle(data_1)
        if len(data_1) == 0:
            d_1 = np.zeros((512, 512, 3))
        else:
            d_1 = np.array(Image.open(data_1[0]))

        data_2 = glob.glob(p_2)
        random.shuffle(data_2)
        if len(data_2) == 0:
            d_2 = np.zeros((512, 512, 3))
        else:
            d_2 = np.array(Image.open(data_2[0]))
        out = {
            '0': self.transform(d_0),
            '1': self.transform(d_1),
            '2': self.transform(d_2)
        }
        label = self.target[idx]
        return out, label


def data_get(data_path='../data'):
    print("load data_patient")
    path_0 = f'{data_path}/normal/train/*'
    patient_normal = glob.glob(path_0)
    random.shuffle(patient_normal)

    path_1 = f'{data_path}/disease/train/*'
    patient_cancer = glob.glob(path_1)
    random.shuffle(patient_cancer)

    patient_all = patient_normal + patient_cancer
    target = [0.] * len(patient_normal) + [1.] * len(patient_cancer)

    print(len(patient_all))
    return patient_all, target


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x, y = data_get()
