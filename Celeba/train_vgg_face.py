import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import classify
import utils
import utils
import load_gaze
from torchsummary import summary
from utils import *
from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_logger, create_loss, create_model,
                             create_optimizer, create_scheduler,
                             create_tensorboard_writer)
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                   create_train_output_dir, load_config,
                                   save_config, set_seeds, setup_cudnn)
from gaze_estimation.config import get_default_config 
from gaze_estimation.datasets import create_dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

config = get_default_config()
config.merge_from_file('configs/mpiifacegaze/resnet_simple_14_train.yaml')
config.freeze()
train_dataset, val_dataset = create_dataset(config, True, [0,2,3,6,8,9,10,11,12,14], True)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.train_dataloader.num_workers,
    pin_memory=config.train.train_dataloader.pin_memory,
    drop_last=config.train.train_dataloader.drop_last,
)



if __name__ == "__main__":
    dataset_name = "gaze"
    file = "./Celeba/" + dataset_name + ".json"
    file = "./gaze.json"

    args = utils.load_params(json_file=file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = args["VGG16"]["batch_size"]
    # batch_size = 64

    model = classify.VGG16(n_classes=15)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args["VGG16"]["lr"], momentum=0.9)

    for epoch in range(args["VGG16"]["epochs"]):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            print(f'{i}/{len(train_loader)}', end='\r')
            images, poses, gazes, ids = data
            # for i in range(len(ids)):
            #     ids[i] = [0,2,3,6,8,9,10,11,12,14].index(ids[i])
            images = images.to(device)
            ids = ids.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs[1], ids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1)}')

        #Validation loop
        # correct = 0
        # total = 0
        # i = 0
        # with torch.no_grad():
        #     for data in testloader:
        #         print(f'{i}/{len(testloader)}', end='\r')
        #         inputs, labels, files = data
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs[1].data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #         i += 1
        # print(f'Accuracy: {correct / total}, wrong count: {total-correct}')

        torch.save(model, './result/gazeFaceClassifier_full.zip')

    print('Finished Training')