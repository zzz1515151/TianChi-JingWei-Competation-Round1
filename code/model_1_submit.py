import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils.utils as utils
from model_1 import *

if not os.path.exists('../submit/result_model_1'):
	os.makedirs('../submit/result_model_1')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

pretrain_path = '../data/jingwei/model_1_exp/model_1_exp_epoch40'
stride = (448, 448)
pretrained_model = torch.load(pretrain_path) 
#################### for multi gpus ############################### 
pretrained_model = utils.parallel_transfer(pretrained_model['network'])
net_config = {}
net_config['num_classes'] = 4
net = DeepLabV3_4(net_config)
if pretrained_model.keys() == net.state_dict().keys():
	net.load_state_dict(pretrained_model)
else:
	raise KeyError('keys do not match! Check you model!')
net = net.eval()
net = net.cuda()

image_transform = transforms.Compose([			
            lambda x: x.astype(np.float32),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], #resnet
                std = [0.229, 0.224, 0.225])
            ])

################################## for image 3 ###########################################
image_path = '../data/jingwei_round1_test_a_20190619/image_3.png'
save_path = '../submit/result_model_1/image_3_predict.png'
print('processing test 3......')
print('loading data......')
Image.MAX_IMAGE_PIXELS = None
image = Image.open(image_path)
image = np.asarray(image)
print('data OK!')
image = image[:, :, 0:3]
h = image.shape[0]
w = image.shape[1]
padding_h = (h // stride[0] + 1) * stride[0]
padding_w = (w // stride[1] + 1) * stride[1]
padding_image = np.zeros([padding_h, padding_w, 3])
padding_image[0:h, 0:w, :] = image
padding_results = np.zeros([padding_h, padding_w])

for i in range(padding_h // stride[0]):
	for j in range(padding_w // stride[1]):
		crop = padding_image[i * stride[0]:(i + 1) * stride[0], j * stride[1]:(j + 1) * stride[1], :]		
		#assert crop.shape[0] == crop.shape[1] == 448
		crop = image_transform(crop)
		crop = crop.unsqueeze(0)
		crop = crop.cuda()
		with torch.no_grad():
			prob_map = net(crop)
		max_map = torch.max(prob_map, dim = 1)[1]
		max_map = max_map.cpu().numpy().squeeze(0)
		padding_results[i * stride[0]:(i + 1) * stride[0], j * stride[1]:(j + 1) * stride[1]] = max_map
		print('[{}]/[{}]'.format(
					i * padding_w // stride[1] + j + 1,
					padding_h // stride[0] * padding_w // stride[1]
								))
padding_results = padding_results[0:h, 0:w]
padding_results = Image.fromarray(padding_results.astype(np.uint8))
padding_results.save(save_path)
print('Saving at: {}'.format(save_path))

################################## for image 4 ###########################################
image_path = '../data/jingwei_round1_test_a_20190619/image_4.png'
save_path = '../submit/result_model_1/image_4_predict.png'
print('processing test 4......')
print('loading data......')
Image.MAX_IMAGE_PIXELS = None
image = Image.open(image_path)
image = np.asarray(image)
print('data OK!')
image = image[:, :, 0:3]
h = image.shape[0]
w = image.shape[1]
padding_h = (h // stride[0] + 1) * stride[0]
padding_w = (w // stride[1] + 1) * stride[1]
padding_image = np.zeros([padding_h, padding_w, 3])
padding_image[0:h, 0:w, :] = image
padding_results = np.zeros([padding_h, padding_w])

for i in range(padding_h // stride[0]):
	for j in range(padding_w // stride[1]):
		crop = padding_image[i * stride[0]:(i + 1) * stride[0], j * stride[1]:(j + 1) * stride[1], :]		
		#assert crop.shape[0] == crop.shape[1] == 448
		crop = image_transform(crop)
		crop = crop.unsqueeze(0)
		crop = crop.cuda()
		with torch.no_grad():
			prob_map = net(crop)
		max_map = torch.max(prob_map, dim = 1)[1]
		max_map = max_map.cpu().numpy().squeeze(0)
		padding_results[i * stride[0]:(i + 1) * stride[0], j * stride[1]:(j + 1) * stride[1]] = max_map
		print('[{}]/[{}]'.format(
					i * padding_w // stride[1] + j + 1,
					padding_h // stride[0] * padding_w // stride[1]
								))
padding_results = padding_results[0:h, 0:w]
padding_results = Image.fromarray(padding_results.astype(np.uint8))
padding_results.save(save_path)
print('Saving at: {}'.format(save_path))
