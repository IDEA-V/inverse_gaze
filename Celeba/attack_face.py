import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torchvision.transforms as transforms
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from gaze_estimation.utils import (AverageMeter, compute_angle_error)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = "cuda"
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def inversion(G, D, T, E, gazes, ids, ground_truth, lr=2e-2, momentum=0.9, lamda=100, iter_times=15000, clip_range=1):
	gazes = gazes.cuda()
	ids = ids.long().cuda()
	criterion = nn.L1Loss(reduction='mean').cuda()
	criterion1 = nn.CrossEntropyLoss().cuda()
	bs = gazes.shape[0]
	
	iden = torch.zeros(10)
	for i in range(10):
		iden[i] = i
	iden = iden.view(-1).long().cuda()

	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)
	rz = transforms.Resize((224,224))

	for random_seed in range(5):
		tf = time.time()
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
			
		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			fake_in = rz(fake)
			out = T(fake_in)
			out1 =  E(fake)

			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = - label.mean()
			# Iden_Loss = criterion(nn.functional.normalize(out), nn.functional.normalize(gazes))
			Gaze_Loss = criterion(out, gazes)
			Iden_Loss = criterion1(out1[1], ids)
			Total_Loss = 0.1*Prior_Loss +  lamda * Gaze_Loss
			# Total_Loss = Prior_Loss +  lamda*Iden_Loss


			print(f"Loss: {Total_Loss.item():.2f} Prior_Loss: {Prior_Loss.item():.2f} Angle Error: {compute_angle_error(out, gazes).mean():.2f}", end='\r')
			# print(f"Loss: {Total_Loss.item():.2f} Prior_Loss: {Prior_Loss.item():.2f} Gaze_Loss: {Gaze_Loss.item():.2f} Iden_Loss: {Iden_Loss.item():2f}", end='\r')


			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				imgs = []
				for j in range(10):
					img = fake_img[j]
					img = torch.concat((img[0].unsqueeze(0).cpu(), ground_truth[j][0].unsqueeze(0)))
					img = img.unsqueeze(1)
					imgs.append(img)

				img = torch.concat(imgs)
				tvls.save_image(img, f'./result/face_gaze/img_{i+1}.png', nrow=2)
			
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(utils.low2high(fake))[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if score[i, i].item() > max_score[i].item():
				max_score[i] = score[i, i]
				max_iden[i] = eval_iden[i]
				z_hat[i, :] = z[i, :]
			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
		
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / 100))

	correct = 0
	for i in range(bs):
		gt = iden[i].item()
		if max_iden[i].item() == gt:
			correct += 1
	
	correct_5 = torch.sum(flag)
	acc, acc_5 = correct * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.2f}\tAcc5:{:.2f}".format(acc, acc_5))
	

if __name__ == '__main__':
	target_path = "./result/gazeEstimater.zip"
	g_path = "./result/models_celeba_gan/celeba_G1.tar"
	d_path = "./result/models_celeba_gan/celeba_D1.tar"
	e_path = "./result/gazeFaceClassifier_full.zip"
	
	T = torch.load(target_path)
	freeze(T)
	T = nn.DataParallel(T).cuda()

	E = torch.load(e_path)
	freeze(E)
	E = nn.DataParallel(E).cuda()

	E1 = torch.load(e_path)
	freeze(E1)
	E1 = nn.DataParallel(E1).cuda()

	G = generator.Generator()
	G = nn.DataParallel(G).cuda()
	ckp_G = torch.load(g_path)['state_dict']
	utils.load_my_state_dict(G, ckp_G)

	D = discri.DGWGAN()
	D = nn.DataParallel(D).cuda()
	ckp_D = torch.load(d_path)['state_dict']
	utils.load_my_state_dict(D, ckp_D)


	config = get_default_config()
	config.merge_from_file('configs/mpiifacegaze/resnet_simple_14_train.yaml')
	config.freeze()
	train_dataset, val_dataset = create_dataset(config, False, [0,2,3,6,8,9,10,11,12,14])

	images = []
	gazes = []
	ids = []
	for d in train_dataset:
		images.append(d[0])
		gazes.append(d[2])
		ids.append(d[3])

	images = torch.cat([image.unsqueeze(0) for image in images])
	gazes = torch.cat([gaze.unsqueeze(0) for gaze in gazes])
	ids = torch.from_numpy(np.array(ids))

	tvls.save_image(images, f'img_gt.png')
	
	inversion(G, D, T, E, gazes, ids, images)