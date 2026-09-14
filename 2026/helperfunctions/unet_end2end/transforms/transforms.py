"""
Contains a function returning a transform that randomly flip images horizontally and
vertically.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torchvision.transforms.v2 as T


def get_random_flip_transform() -> T.Compose:
    trans = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
    return trans
