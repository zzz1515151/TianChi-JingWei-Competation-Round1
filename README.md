# jingwei-round1-submit (team *testB*)

## Introduction
This is a submission code of team *testB* in [Tianchi Competition: Jingwei], achieving **0.3484** on leaderboard. We use two models to get the result above, here is the implementation guide.

## File structure
The file structure, following jingwei submission code standard, is as follow:

jingwei-round1-submit  
|--README.md  
|--submit
|   |--result_model_2    
|   |--result_model_1    
|   |--result_final             
|--data   
|   |--jingwei   
|   |--train_model_1  
|   |--train_model_2 
|   |--jingwei_round1_train_20190619.zip  
|   |--jingwei_round1_test_a_20190619.zip  
|--code   
|   |--config   
|   |--dataloader  
|   |--model_1   
|   |--model_2   
|   |--utils    

./data/jingwei contains ckpt and log file while training.        
./data/train_model_1 is the training data for model_1.  
./data/train_model_2 is the training data for model_2.         

./code/config contains training config file.  
./code/dataloader contains datasets and data augmentation codes.    
./code/model_1 and ./code/model_2 contains the two models we use for competition.  

./submit/result_model_1 contains test results of model_1.           
./submit/result_model_2 contains test results of model_2.           
./submit/result_final contains out final results.           

## Implementation of model_1

### Dependencies
We use pytorch framework for training, before training, use requirement.txt to install dependencies.

`cd code`       
`pip install -r requirements_1.txt`

We also use pretrained resnet50 weight file from https://download.pytorch.org/models/resnet50-19c8e357.pth. The file should be placed under ./data 

Three NVIDIA-2080Ti GPUs and 128G RAM are used while training.

### Creating training set 
We generate our own training set from the official one. To obtain our training set, run
`python create_model_1_train.py`        
This would create a train_model_1 dir under ./data

### Train and submit
To train model_1, run       
`python model_1_main.py --exp model_1_exp`          
To get predictions, run                 
`python model_1_submit.py`          
Or simply run       
`./model_1_predict.sh`          

Until now you should be able to reproduce our result of model_1. Achieving test acc of 0.3230.

## Implementation of model_2

### Dependencies
We use pytorch framework for training, before training, use requirement.txt to install dependencies.

`cd code/model_2`       
`pip install -r requirements_2.txt`

We also use pretrained resnet34 weight file from https://download.pytorch.org/models/resnet34-333f7ec4.pth. The file should be placed under ./data 

Three NVIDIA-1080Ti GPUs and 64G RAM are used while training.

## Train and submit
First move to model_2 directory:
`cd code/model_2`   
To train model_2, run       
`CUDA_VISIBLE_DEVICES=0,1,2 python main.py --backbone resnet --lr 0.007 --workers 12 --epochs 50 --batch-size 12 --gpu-ids 0,1,2 --checkname UNetResNet34 --eval-interval 1 --dataset jingwei --model-name UNetResNet34 --pretrained --base-size 600 --crop-size 512 --loss-type focal`          
To get predictions, run                 
`python model_2_submit_1.py`         
and
`python model_2_submit_2.py`     
  

Until now you should be able to reproduce our result of model_2. Achieving test acc of 0.3034.


## Combination of model_1 and model_2
To get out final submit, run   
`python final_result.py`            
You can find our final results at ./submit/result_final             






