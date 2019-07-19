import os
import cv2
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

if not os.path.exists('../data/train_model_1'):
	os.makedirs('../data/train_model_1')

count = 0
########################### for image 1 ############################################

print('processing image 1 ......')

img = Image.open('../data/jingwei_round1_train_20190619/image_1.png')   
img = np.asarray(img)

anno_map = Image.open('../data/jingwei_round1_train_20190619/image_1_label.png')   
anno_map = np.asarray(anno_map)

size = 448
stride = 224
h = img.shape[0]
w = img.shape[1]

h_step = (h - size) // stride + 1
w_step = (w - size) // stride + 1

for i in range(h_step):
	for j in range(w_step):		
		crop_image = img[i * stride:i * stride + size, j * stride:j * stride + size]
		crop_label = anno_map[i * stride:i * stride + size, j * stride:j * stride + size]
		if 255 in crop_image[:, :, -1]:
			image_path = '../data/train_model_1/{:0>5d}.png'.format(count)
			label_path = '../data/train_model_1/{:0>5d}_label.png'.format(count)
			crop_image = crop_image[:, :, 0:3]
			crop_image = crop_image[:, :, [2, 1, 0]]
			cv2.imwrite(image_path, crop_image)
			cv2.imwrite(label_path, crop_label)
			count += 1
		print('[{}]/[{}]'.format(i * w_step + j, w_step * h_step))

print('finish image 1 with {} images'.format(count))

########################### for image 2 ############################################

print('processing image 2 ......')

img = Image.open('../data/jingwei_round1_train_20190619/image_2.png')   
img = np.asarray(img)

anno_map = Image.open('../data/jingwei_round1_train_20190619/image_2_label.png')   
anno_map = np.asarray(anno_map)

size = 448
stride = 224
h = img.shape[0]
w = img.shape[1]

h_step = (h - size) // stride + 1
w_step = (w - size) // stride + 1

for i in range(h_step):
	for j in range(w_step):		
		crop_image = img[i * stride:i * stride + size, j * stride:j * stride + size]
		crop_label = anno_map[i * stride:i * stride + size, j * stride:j * stride + size]
		if 255 in crop_image[:, :, -1]:
			image_path = '../data/train_model_1/{:0>5d}.png'.format(count)
			label_path = '../data/train_model_1/{:0>5d}_label.png'.format(count)
			crop_image = crop_image[:, :, 0:3]
			crop_image = crop_image[:, :, [2, 1, 0]]
			cv2.imwrite(image_path, crop_image)
			cv2.imwrite(label_path, crop_label)
			count += 1
		print('[{}]/[{}]'.format(i * w_step + j, w_step * h_step))
		
print('finish creating train data with total {} images'.format(count))

############################## create csv ##########################################################
print('creating csv')
with open('../data/train_model_1/train_model_1.csv', 'w') as f:
	for i in range(count):
		line = '../data/train_model_1/{:0>5d}.png ../data/train_model_1/{:0>5d}_label.png\n'.format(i, i)
		f.write(line)

print('finish creating csv')