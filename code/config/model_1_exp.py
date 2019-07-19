config = {}

train_config = {}
train_config['csv_root'] = '../data/train_model_1/train_model_1.csv'
train_config['batch_size'] = 48
config['train_config'] = train_config

net = {}
net['num_classes'] = 4
config['net'] = net

optim = {}
optim["lr"] = 1e-3
optim["momentum"] = 0.9
optim["weight_decay"] = 5e-4
optim["nesterov"] = True
config['optim'] = optim
############## global config #########################
config['display_step'] = 1
config['num_epochs'] = 50

