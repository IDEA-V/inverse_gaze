import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from discri import DGWGAN
from generator import Generator, InversionNet

import torchvision.utils
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
from losses import noise_loss
import matplotlib.pyplot as plt

config = get_default_config()
config.merge_from_file('configs/mpiifacegaze/resnet_simple_14_train.yaml')
config.freeze()
train_dataset, val_dataset = create_dataset(config, True, [1,4,5,7,13], False, True)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.val_dataloader.num_workers,
    pin_memory=config.train.train_dataloader.pin_memory,
    drop_last=config.train.train_dataloader.drop_last,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.train.batch_size,
    shuffle=False,
    num_workers=config.train.val_dataloader.num_workers,
    pin_memory=config.train.val_dataloader.pin_memory,
    drop_last=False,
)

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

save_img_dir = "result/imgs_celeba_gan"
save_model_dir= "result/models_celeba_gan"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "GAN.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')

if __name__ == "__main__":
    
    file = "./" + dataset_name + ".json"
    args = load_params(json_file=file)

    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']
    rz = transforms.Resize((224,224))
    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    # dataset, dataloader = init_gaze_face_data('./data/MPIIFaceGaze_normalized', [1,4,5,7,13])
    target_path = "./result/gazeEstimater.zip"
    T = torch.load(target_path)
    Net = InversionNet()
    DG = DGWGAN(3, 32)
    
    T = nn.DataParallel(T).cuda()
    Net = torch.nn.DataParallel(Net).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(Net.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, (images,blurred,poses, gazes) in enumerate(train_loader):
            step += 1
            # imgs = images.cuda()
            imgs = images.cuda()
            blurred = blurred.cuda()
            bs = imgs.size(0)
            
            freeze(Net)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = Net((blurred, z))

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0
            

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            if step % n_critic == 0:
                # train G
                freeze(DG)
                unfreeze(Net)
                z1 = torch.randn(bs, z_dim).cuda()
                output1 = Net((blurred, z1))

                z2 = torch.randn(bs, z_dim).cuda()
                output2 = Net((blurred, z2))

                logit_dg = DG(output1)
                diff_loss = noise_loss(T, rz(output1), rz(output2))

                # calculate g_loss
                g_loss = - logit_dg.mean() - diff_loss * 0.5
                print(f'{i}/{len(train_loader)} dg_loss: {dg_loss} g_loss: {g_loss}', end='\r')
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        print("Epoch:%d \t Time:%.2f" % (epoch, interval))
        if (epoch+1) % 1 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = Net((blurred, z))
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image1_{}.png".format(epoch)), nrow = 8)
        
        torch.save({'state_dict':Net.state_dict()}, os.path.join(save_model_dir, "celeba_G_auxiliary.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "celeba_D_auxiliary.tar"))

