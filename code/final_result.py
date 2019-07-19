from PIL import Image 
import numpy as np 
Image.MAX_IMAGE_PIXELS = None 

if not os.path.exists('../submit/result_final'):
	os.makedirs('../submit/result_')

#####################################image 3#####################################

main_res_path = '../submit/result_model_2/image_3_predict.png'
sup_res_path = '../submit/result_model_1/image_3_predict.png'
save_path = '../submit/result_final/image_3_predict.png'

main_image = Image.open(main_res_path)
sup_image = Image.open(sup_res_path)
print('stage_1 done!')

main_image = np.asarray(main_image)
sup_image = np.asarray(sup_image)
print('stage_2 done!')
main_none = np.where(main_image == 0)
print('stage_3 done!')
main_image.flags.writeable = True
main_image[main_none] = sup_image[main_none]
print('stage_4 done!')

main_image = Image.fromarray(main_image.astype(np.uint8))
main_image.save(save_path)
print('save as {}'.format(save_path))

#####################################image 4#####################################

main_res_path = '../submit/result_model_2/image_4_predict.png'
sup_res_path = '../submit/result_model_1/image_4_predict.png'
save_path = '../submit/result_final/image_4_predict.png'

main_image = Image.open(main_res_path)
sup_image = Image.open(sup_res_path)
print('stage_1 done!')

main_image = np.asarray(main_image)
sup_image = np.asarray(sup_image)
print('stage_2 done!')
main_none = np.where(main_image == 0)
print('stage_3 done!')
main_image.flags.writeable = True
main_image[main_none] = sup_image[main_none]
print('stage_4 done!')

main_image = Image.fromarray(main_image.astype(np.uint8))
main_image.save(save_path)
print('save as {}'.format(save_path))