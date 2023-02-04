import pathlib
from typing import List, Union

import torch
import yacs.config
from torch.utils.data import Dataset

from ..transforms import create_transform
from ..types import GazeEstimationMethod


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True, public:List = list(range(15)), face:bool = False) -> Union[List[Dataset], Dataset]:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        from .mpiigaze import OnePersonDataset
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        from .mpiifacegaze import OnePersonDataset
    else:
        raise ValueError

    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    assert dataset_dir.exists()
    assert config.train.test_id in range(-1, 15)
    assert config.test.test_id in range(15)
    person_ids = [f'p{index:02}' for index in public]

    transform = create_transform(config)

    if is_train:
        if config.train.test_id == -1:
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform, face)
                for person_id in person_ids
            ])
            # assert len(train_dataset) == 45000
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform, face)
                for person_id in person_ids if person_id != test_person_id
            ])
            # assert len(train_dataset) == 42000

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        images = []
        poses = []
        gazes = []
        ids = []

        train_dataset = torch.utils.data.ConcatDataset([
            OnePersonDataset(person_id, dataset_dir, transform, True, 1)
        for person_id in person_ids])

        return torch.utils.data.dataset.random_split(train_dataset, [len(person_ids), 0])

        for p in person_ids:
            d = OnePersonDataset(p, dataset_dir, transform, True, 10)
            images.append(d[0])
            poses.append(d[1])
            gazes.append(d[2])
            ids.append(int(p[1:]))

        images = torch.cat([image.unsqueeze(0) for image in images])
        gazes = torch.cat([gaze.unsqueeze(0) for gaze in gazes])
        poses = torch.cat([pose.unsqueeze(0) for pose in poses])

        return images, poses, gazes, ids