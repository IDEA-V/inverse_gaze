from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import yacs.config

from .types import GazeEstimationMethod


def create_transform(config: yacs.config.CfgNode) -> Any:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        return _create_mpiigaze_transform(config)
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        return _create_mpiifacegaze_transform(config)
    else:
        raise ValueError

def _scale(x):
    return x.astype(np.float32) / 255

def _index(x):
    return x[None, :, :]

def _unsqueeze(x):
    return x.unsqueeze(0)

def _squeeze(x):
    return x.squeeze()

def _create_mpiigaze_transform(config: yacs.config.CfgNode) -> Any:
    # transform = torchvision.transforms.Compose([
    #     _scale,
    #     torch.from_numpy,
    #     _unsqueeze,
    #     torchvision.transforms.Resize((64, 64)),
    #      _squeeze,
    #     _index,
    # ]) 
    # return transform
    proc = []
    proc.append(torchvision.transforms.ToPILImage())
    proc.append(torchvision.transforms.Resize((64, 64)))
    proc.append(torchvision.transforms.ToTensor())
        
    return torchvision.transforms.Compose(proc)

def _resize(x):
    return cv2.resize(x, (64, 64))

def _scale(x):
    return x.astype(np.float32) / 255

def _to_gray(x):
    return cv2.cvtColor(
            cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.
            COLOR_GRAY2BGR)

def _transpose(x):
    return x.transpose(2, 0, 1)

def _create_mpiifacegaze_transform(config: yacs.config.CfgNode) -> Any:

    transform = torchvision.transforms.Compose([
        _resize,
        _to_gray,
        _transpose,
        _scale,
        torch.from_numpy,
        # torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
        #                                  std=[0.225, 0.224, 0.229]),
    ])
    return transform
