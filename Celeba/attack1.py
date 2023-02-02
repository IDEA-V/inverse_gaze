import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torchvision.transforms as transforms
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from gaze_estimation.utils import (AverageMeter, compute_angle_error)
                                   
device = "cuda"
num_classes = 1000
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)

def inversion(G, D, T, E, poses, gazes, lr=2e-2, momentum=0.9, lamda=100, iter_times=15000, clip_range=1):
	gazes = gazes.cuda()
	criterion = nn.MSELoss(reduction='mean').cuda()
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
			fake_in = fake[:,0,:,:].unsqueeze(1)
			out = T(fake_in, poses)
			
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = - label.mean()
			Iden_Loss = criterion(nn.functional.normalize(out), nn.functional.normalize(gazes))
			# Iden_Loss = compute_angle_error(out, gazes).mean()
			Total_Loss = Prior_Loss +  Iden_Loss

			print(f"Loss: {Total_Loss.item():.2f} Prior_Loss: {Prior_Loss.item():.2f} L2_Loss: {compute_angle_error(out, gazes).mean():.2f}", end='\r')

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			rz = transforms.Resize((36, 60))
			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				for i in range(10):
					img = rz(fake_img[i])
					tvls.save_image(img, f'img_{i}.png')
				eval_prob = E(fake_img)[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
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
	e_path = "./result/gazeClassifier.zip"
	
	# T = classify.VGG16(10)
	# T = nn.DataParallel(T).cuda()
	# ckp_T = torch.load(target_path)['state_dict']
	# utils.load_my_state_dict(T, ckp_T)
	T = torch.load(target_path)
	T = nn.DataParallel(T).cuda()

	E = torch.load(e_path)
	E = nn.DataParallel(E).cuda()

	G = generator.Generator()
	G = nn.DataParallel(G).cuda()
	ckp_G = torch.load(g_path)['state_dict']
	utils.load_my_state_dict(G, ckp_G)

	D = discri.DGWGAN()
	D = nn.DataParallel(D).cuda()
	ckp_D = torch.load(d_path)['state_dict']
	utils.load_my_state_dict(D, ckp_D)

	config = get_default_config()
	config.merge_from_file('configs/mpiigaze/lenet_train.yaml')
	config.freeze()
	poses, gazes = create_dataset(config, False, [0,2,3,6,8,9,10,11,12,14])

	inversion(G, D, T, E, poses, gazes)