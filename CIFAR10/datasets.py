import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T


from PIL import Image



"""
DATASET CODE
"""
class RestrictedClassDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset, classes=None, class_map=None):
        super().__init__(dataset.root, transform=dataset.transform, target_transform=dataset.target_transform)

        classes_was_none = classes == None
        if classes_was_none:
            classes = list(set(dataset.targets))
        self.classes = classes

        if class_map != None and not classes_was_none: # only map if classes were provided
            classes = [class_map[c] for c in classes]

        # convert dataset.targets to tensor
        tensor_targets = torch.tensor(dataset.targets)
        class_mask = sum(tensor_targets == class_ for class_ in classes).bool()
        self.data = dataset.data[class_mask]
        self.targets = tensor_targets[class_mask]
        
        target_copy = copy.deepcopy(self.targets)
        for target in classes:
            self.targets[target_copy == target] = torch.tensor(classes.index(target))

        print('dataset size', len(self.data), len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def dataset_class_mapper(dataset, classes):
    return RestrictedClassDataset(dataset, classes=classes, class_map={
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }) # cifar map


"""
TRANSFORM CODE
"""



normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])

class ContrastiveLearningTransform:
    def __init__(self):
        transforms = [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ]

        self.transform = T.Compose(transforms)

    def __call__(self, x):
        out = [single_transform(self.transform(x)), single_transform(self.transform(x))]
        return out

class SingleTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        out = [single_transform(x), single_transform(x)]
        return out

def rotate_images(images):
    nimages = images.shape[0]
    n_rot_images = 4 * nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda()
    rot_classes = torch.zeros([n_rot_images]).long().cuda()

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages:2 * nimages] = images.flip(3).transpose(2, 3)
    rot_classes[nimages:2 * nimages] = 1
    # rotate 180
    rotated_images[2 * nimages:3 * nimages] = images.flip(3).flip(2)
    rot_classes[2 * nimages:3 * nimages] = 2
    # rotate 270
    rotated_images[3 * nimages:4 * nimages] = images.transpose(2, 3).flip(3)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes