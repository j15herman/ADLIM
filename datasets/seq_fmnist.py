# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sequential Fashion-MNIST: a drop-in, harder-features sibling of Sequential
MNIST for cross-dataset validation. Identical continual protocol -- 5 tasks x 2
classes, class-il, the same `mnistmlp` backbone, 28x28 grayscale, cross-entropy
-- so every plasticity_mlp mechanism (per-weight metaplasticity, budget, dendrite
geometry, per-class-block head consolidation) applies unchanged; only the source
images differ. Use it to check that a Seq-MNIST result transfers to a genuinely
harder image distribution without touching the model."""

from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import FashionMNIST

from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args


class MyFashionMNIST(FashionMNIST):
    """
    Overrides the FashionMNIST dataset to change the getitem function (mirrors
    MyMNIST: also return the un-augmented image, and logits when distilling).
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyFashionMNIST, self).__init__(root, train,
                                             transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img


class SequentialFMNIST(ContinualDataset):
    """The Sequential Fashion-MNIST dataset (same protocol as Sequential MNIST).

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
    """

    NAME = 'seq-fmnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (28, 28)
    TRANSFORM = None

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = transforms.ToTensor()
        train_dataset = MyFashionMNIST(base_path() + 'FashionMNIST',
                                       train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(base_path() + 'FashionMNIST',
                                    train=False, download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "mnistmlp"

    @staticmethod
    def get_transform():
        return SequentialFMNIST.TRANSFORM

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 64

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 1

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        # FashionMNIST.classes are already clean labels ("T-shirt/top", "Trouser",
        # ...) -- no "N - Name" prefix to strip (unlike MNIST).
        classes = FashionMNIST(base_path() + 'FashionMNIST', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
