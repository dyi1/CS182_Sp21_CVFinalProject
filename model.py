import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import os 
import numpy as np
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):
        super(Net, self).__init__()
        # resnet18 = models.resnet18(pretrained=True)
        # resnet18.fc = nn.Linear(7*7*512, 200)

        self.layer1 = nn.Linear(im_height * im_width * 3, 128)
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class ValFolder(torchvision.datasets.ImageFolder):
    lines = None
    CLASS_NAMES = np.array([])

    def __init__(self, CLASS_NAMES, *args):
        super(ValFolder, self).__init__(*args)
        self.CLASS_NAMES = CLASS_NAMES

    def __getitem__(self, index):
        if self.lines is None:
            with open(os.path.join(self.root, "val_annotations.txt")) as f:
                self.lines = f.readlines()
        line = self.lines[index].split()
        sample = self.loader(os.path.join(self.root, "images", line[0]))
        target = np.where(self.CLASS_NAMES == line[1])[0][0]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target
