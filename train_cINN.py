import os
from cINN.model import FullModel as cINN
import torch
import dill
import sys
from datetime import datetime

# Load samples
savefile = './Samples/samples.npz'
with open(savefile, mode='rb') as file:
    (In,Out, ch_in, ch_in_tex) = dill.load(file)

# Reduce to sensitivity selected sub space
feat = np.array([  1,   2,   3,   5,   6,   7,   8,  30,  31,  32,  33,  43,  44,
         45,  46,  47,  48,  53,  61,  62,  63,  64,  65,  66,  67,  68,
         69,  70,  71,  72,  73,  74,  80,  81,  85,  86,  87,  88,  91,
         92,  93, 101, 122, 138, 139, 140, 147, 148, 149])
In = In[:, feat]
ch_in = ch_in[:, feat]
ch_in_tex = ch_in_tex[:, feat]



# Creat log folder and file
case_name = 'Beam'
now = datetime.now()
folder = './Current_Training/' + now.strftime('%Y-%m-%d-%H-%M_') + case_name + '/'
try:
    os.mkdir('./Current_Training/')
except:
    pass

try:
    os.mkdir(folder)
except:
    pass

cout = sys.stdout = open(folder + 'progress.txt', 'wb')

#Initialize hyper parameter set and define the hyperparameters
hps = cINN.default_hps()
hps['feat_in'] = In.shape[1]
hps['feat_cond'] = (Out.shape[2], Out.shape[1])

# Define the cINN architecture
hps['n_CCB'] = 15
hps['condnet_out_dim'] = [100, 200, 300, 400, 500] #[200, 300, 400, 500, 600] #[100, 200, 300, 400, 500]  #np.cumsum(np.ones(3, dtype=int)*100)+100 #Out.shape[1]*Out.shape[2]
hps['condnet_cluster'] = 3
hps['condnet_pooling'] = True
hps['condnet_inter_dim_pow'] = 8 #8#9
hps['subnet_depth'] = 1
hps['subnet_dim'] = [400]*3 + [500]*3 + [600]*3 + [700]*3 + [800]*3 #900 #np.cumsum(np.ones(hps['n_CCB'], dtype=int)*20)+300
hps['batch_size'] = 64
hps['subnet_dropout'] = False
hps['subnet_dropout_rate'] = 0.125
hps['subnet_batch_norm'] = True

# Define the training parameters
hps['epochs'] = 150
hps['learning_rate'] = 0.3
hps['lr_scheduler'] = True
hps['lr_epoch_step'] = 10
hps['lr_factor'] = 0.75
hps['optimizer'] = 'adagrad'
hps['grad_clip'] = 0.01
hps['test_split'] = 0.1

# Create the cINN
model = cINN(hps=hps, ch_name_tex=ch_in_tex, ch_name=ch_in)

# Transform the samples to tensors
In = torch.tensor(In, dtype=torch.float, device=model.hps['device'])
Out = torch.tensor(Out, dtype=torch.float, device=model.hps['device'])

# Train the cINN
model.train_model(In, Out, check_int=10, check_dir=folder, save_file=folder + 'Model_trained.pt')

# Output the final training progress
model.print_training(fname=folder + 'Training.png', dpi=300, figsize=[15, 12])