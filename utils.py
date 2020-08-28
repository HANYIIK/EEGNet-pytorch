from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

'''

Train Dataset:

(1440, 64, 256)
apple       180
banana      180
lemon       180
watermelon  180
lion        180
tiger       180
wolf        180
leopard     180
'''

'''

Test Dataset:

(160, 64, 256)
apple       20
banana      20
lemon       20
watermelon  20
lion        20
tiger       20
wolf        20
leopard     20
'''
PATH = 'data/'

# LABELS
# 2 classes
def get_2_classes_labels():
    fruits_train_labels = np.ones(720, dtype=np.int)
    animals_train_labels = np.zeros(720, dtype=np.int)

    fruits_test_labels = np.ones(80, dtype=np.int)
    animals_test_labels = np.zeros(80, dtype=np.int)

    return torch.from_numpy(np.hstack((fruits_train_labels, animals_train_labels))), torch.from_numpy(np.hstack((fruits_test_labels, animals_test_labels)))

# 4 classes
def get_4_classes_labels():
    train_labels = np.zeros(720, dtype=np.int)
    train_labels[180:360] = 1
    train_labels[360:540] = 2
    train_labels[540:720] = 3

    test_labels = np.zeros(80, dtype=np.int)
    test_labels[20:40] = 1
    test_labels[40:60] = 2
    test_labels[60:80] = 3

    return torch.from_numpy(train_labels), torch.from_numpy(test_labels)

# 8 classes
def get_8_classes_labels():
    train_labels = np.zeros(1440, dtype=np.int)
    train_labels[180:360] = 1
    train_labels[360:540] = 2
    train_labels[540:720] = 3
    train_labels[720:900] = 4
    train_labels[900:1080] = 5
    train_labels[1080:1260] = 6
    train_labels[1260:] = 7

    test_labels = np.zeros(160, dtype=np.int)
    test_labels[20:40] = 1
    test_labels[40:60] = 2
    test_labels[60:80] = 3
    test_labels[80:100] = 4
    test_labels[100:120] = 5
    test_labels[120:140] = 6
    test_labels[140:160] = 7

    return torch.from_numpy(train_labels), torch.from_numpy(test_labels)

# DATASET
def get_dataset():
    fruits_train_arr, fruits_test_arr = get_fruits(dataset=False)
    animals_train_arr, animals_test_arr = get_animals(dataset=False)
    return torch.unsqueeze(torch.from_numpy(np.vstack((fruits_train_arr, animals_train_arr))).type(torch.float32), dim=1), \
           torch.unsqueeze(torch.from_numpy(np.vstack((fruits_test_arr, animals_test_arr))).type(torch.float32), dim=1)

# for 4 classes dataset
def get_fruits(dataset=False):
    apple = resort(loadmat(PATH + 'fruit/apple.mat')['apple'])
    banana = resort(loadmat(PATH + 'fruit/banana.mat')['banana'])
    lemon = resort(loadmat(PATH + 'fruit/lemon.mat')['lemon'])
    watermelon = resort(loadmat(PATH + 'fruit/watermelon.mat')['watermelon'])

    fruits_train = np.vstack((
        np.vstack((apple[:180, :, :], banana[:180, :, :])),
        np.vstack((lemon[:180, :, :], watermelon[:180, :, :]))
    ))
    fruits_test = np.vstack((
        np.vstack((apple[180:, :, :], banana[180:, :, :])),
        np.vstack((lemon[180:, :, :], watermelon[180:, :, :]))
    ))

    if not dataset:
        return fruits_train, fruits_test
    else:
        return torch.unsqueeze(torch.from_numpy(fruits_train).type(torch.float32), dim=1), \
               torch.unsqueeze(torch.from_numpy(fruits_test).type(torch.float32), dim=1)

def get_animals(dataset=False):
    lion = resort(loadmat(PATH + 'animal/lion.mat')['lion'])
    tiger = resort(loadmat(PATH + 'animal/tiger.mat')['tiger'])
    wolf = resort(loadmat(PATH + 'animal/wolf.mat')['wolf'])
    leopard = resort(loadmat(PATH + 'animal/leopard.mat')['leopard'])

    animals_train = np.vstack((
        np.vstack((lion[:180, :, :], tiger[:180, :, :])),
        np.vstack((wolf[:180, :, :], leopard[:180, :, :]))
    ))
    animals_test = np.vstack((
        np.vstack((lion[180:, :, :], tiger[180:, :, :])),
        np.vstack((wolf[180:, :, :], leopard[180:, :, :]))
    ))

    if not dataset:
        return animals_train, animals_test
    else:
        return torch.unsqueeze(torch.from_numpy(animals_train).type(torch.float32), dim=1), \
               torch.unsqueeze(torch.from_numpy(animals_test).type(torch.float32), dim=1)

# reshape from (64, 256, 200) to (200, 64, 256)
def resort(arr):
    return np.array([arr[..., item] for item in range(arr.shape[2])])