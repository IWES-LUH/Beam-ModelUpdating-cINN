import torch
import torch.nn as nn
import torch.optim

import numpy as np
from time import time
import math
import os
from hyperopt import hp

from ray import tune
from cINN.StandardScaler import StandardScaler as stdScaler

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class CondNet(nn.Module):
    '''conditioning network'''

    def __init__(self, n_level, cluster, in_shape, inter_dim_pow, cond_dim, activation, pooling):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()

            def forward(self, x):
                return x.view(x.shape[0], -1)

        n_feat = in_shape[0]
        n_sec = in_shape[1]
        self.cluster = cluster
        self.n_level = n_level
        self.n_steps = int(np.ceil(n_level / cluster))

        # define conv dimensions for intermediate layers
        inter_dim = np.power(2, np.arange(1, inter_dim_pow + 1), dtype=int)
        inter_dim = inter_dim[inter_dim > n_sec]
        if len(inter_dim) >= self.n_steps:
            inter_dim = inter_dim[-(self.n_steps - 1):]
        else:
            inter_dim = np.append(inter_dim, np.ones(self.n_steps - len(inter_dim) - 1) * inter_dim[-1])
        inter_dim = np.append(n_sec, inter_dim)

        dim_out = n_feat * n_sec
        self.levels = nn.ModuleList([None])

        temp_net = nn.Sequential(Flatten())
        temp_net.add_module('cond_1_lin', nn.Linear(int(dim_out), int(cond_dim[0])))

        self.flatten = nn.ModuleList([temp_net])

        for i in range(1, self.n_steps, 1):
            temp_net = nn.Sequential()

            temp_net.add_module('cond_%d_conv1d' % (i + 1),
                                nn.Conv1d(int(inter_dim[i - 1]), int(inter_dim[i]), kernel_size=3, padding=1))
            temp_net.add_module('cond_%d_activ' % (i + 1), activation[i])

            self.levels.append(temp_net)

            if pooling:
                tmp = [nn.AvgPool1d(2, stride=2), Flatten()]
                dim_out = np.floor(n_feat / 2) * inter_dim[i]
            else:
                tmp = [Flatten()]
                dim_out = n_feat * inter_dim[i]

            temp_net = nn.Sequential(*tmp)
            temp_net.add_module('cond_%d_lin' % (i + 1), nn.Linear(int(dim_out), int(cond_dim[i])))

            self.flatten.append(temp_net)

    def forward(self, c):
        intermediate = [c]
        outputs = [self.flatten[0](c)]
        for i in range(1, len(self.levels), 1):
            intermediate.append(self.levels[i](intermediate[-1]))
            outputs.append(self.flatten[i](intermediate[-1]))

        return outputs


class FullModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Set given hyper parameters or generate default values
        if 'hps' in kwargs:
            self.hps = kwargs.get("hps")
        else:
            self.hps = self.default_hps()

        if 'ch_name_tex' in kwargs and len(kwargs.get("ch_name_tex")) == self.hps['feat_in']:
            self.ch_name_tex = kwargs.get("ch_name_tex")
        else:
            self.ch_name_tex = ['ch$_%d$' % (i) for i in range(self.hps['feat_in'])]

        if 'ch_name' in kwargs and len(kwargs.get("ch_name")) == self.hps['feat_in']:
            self.ch_name = kwargs.get("ch_name")
        else:
            self.ch_name = ['ch_%d' % (i) for i in range(self.hps['feat_in'])]

        if self.hps['subnet_dropout']:
            self.hps['subnet_batch_norm'] = False

        self.condnet = self.build_condnets()
        self.condnet.to(self.hps['device'])
        self.cinn = self.build_inn()
        self.cinn.to(self.hps['device'])

        # Save in- and output scalers
        self.in_scaler = stdScaler()
        self.cond_scaler = stdScaler()

        self.getParList()

        for p in self.trainable_parameters:
            p.data = 0.02 * torch.randn_like(p)

        self.optimizer = self.getOptimizer()

    def build_inn(self):

        nodes = [Ff.InputNode(self.hps['feat_in'], name='InputNode')]

        n_cond = self.condnet.n_steps

        c_dim = np.array(self.hps['condnet_out_dim'])
        c_idx = np.repeat(np.arange(n_cond, dtype=int), self.hps['condnet_cluster'])

        # generate conditions
        conditions = [Ff.ConditionNode(c_dim[i], name='CondNode_%d' % i) for i in range(self.condnet.n_steps)]

        if isinstance(self.hps['subnet_dim'], int):
            snet_dim = np.ones(self.hps['n_CCB'], dtype=int) * self.hps['subnet_dim']
        elif len(self.hps['subnet_dim']) != self.hps['n_CCB']:
            raise ValueError('Please assure that subnet_dim is a scalar or vector of size %d' % (self.hps['n_CCB']))
        else:
            snet_dim = self.hps['subnet_dim']
        self.hps['subnet_dim'] = snet_dim

        if isinstance(self.hps['subnet_depth'], int):
            snet_depth = np.ones(self.hps['n_CCB'], dtype=int) * self.hps['subnet_depth']
        elif len(self.hps['subnet_depth']) != self.hps['n_CCB']:
            raise ValueError('Please assure that subnet_depth is a scalar or vector of size %d' % (self.hps['n_CCB']))
        else:
            snet_depth = self.hps['subnet_depth']
        self.hps['subnet_depth'] = snet_depth

        # Chain demanded number of coupling blocks
        for i in range(self.hps['n_CCB']):
            subnet = lambda cin, cout: self.__sub_fc(cin, cout, i)
            dict_CB = {'subnet_constructor': subnet, 'affine_clamping': self.hps['clamp'],
                       'permute_soft': self.hps['permute_soft']}
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.AllInOneBlock,
                                 dict_CB,
                                 conditions=conditions[c_idx[i]],
                                 name=F'coupling_{i}'))

        return Ff.GraphINN(nodes + conditions + [Ff.OutputNode(nodes[-1])], verbose=False)

    def build_condnets(self):

        n = int(np.ceil(self.hps['n_CCB'] / self.hps['condnet_cluster']))

        if isinstance(self.hps['condnet_out_dim'], int):
            out_dim = np.ones(n, dtype=int) * self.hps['condnet_out_dim']
        elif len(self.hps['condnet_out_dim']) == 1:
            out_dim = np.ones(n, dtype=int) * self.hps['condnet_out_dim'][0]
        elif len(self.hps['condnet_out_dim']) < n:
            raise ValueError(
                'Please assure that condnet_out_dim is a scalar or vector of size %d' % (self.hps['n_CCB']))
        else:
            out_dim = self.hps['condnet_out_dim'][:n]
        self.hps['condnet_out_dim'] = out_dim

        if not isinstance(self.hps['condnet_activation'], list):
            activation = [self.hps['condnet_activation'] for i in range(n)]
        elif len(self.hps['condnet_activation']) == 1:
            activation = [self.hps['condnet_activation'][0] for i in range(n)]
        elif len(self.hps['condnet_activation']) < n:
            raise ValueError(
                'Please assure that activation is a single module or list of size %d' % (self.hps['n_CCB']))
        else:
            activation = self.hps['condnet_activation'][:n]
        self.hps['condnet_activation'] = activation

        return CondNet(self.hps['n_CCB'], self.hps['condnet_cluster'], self.hps['feat_cond'],
                       self.hps['condnet_inter_dim_pow'],
                       out_dim, activation, self.hps['condnet_pooling'])

    def forward(self, x, c):
        cond = self.condnet.forward(c)
        return self.cinn.forward(x, c=cond, rev=False, jac=True)

    def reverse(self, z, c):
        self.cinn.eval()
        self.condnet.eval()
        with torch.no_grad():
            cond = self.condnet.forward(c)
            (in_pred, j) = self.cinn.forward(z, c=cond, rev=True, jac=False)
            return in_pred

    def reverse_sample_norm(self, n_rand, c, c_noise=None):
        self.cinn.eval()
        self.condnet.eval()

        if c_noise:
            c = c.clone() * (torch.randn_like(c, dtype=torch.float32, device=self.hps['device']) * c_noise + 1)

        with torch.no_grad():
            n = c.shape[0]

            return [
                self.reverse(torch.randn((n_rand, self.hps['feat_in']), dtype=torch.float32, device=self.hps['device']),
                             c=c[i, :, :].repeat(n_rand, 1, 1).clone()) for i in range(n)]

    def reverse_sample_norm_std_mean(self, n_rand, c, c_noise=None):
        n = c.shape[0]
        in_pred = torch.zeros((n_rand, n, self.hps['feat_in']), dtype=torch.float32, device='cpu')
        if c_noise:
            c = c.clone() * (torch.randn_like(c, dtype=torch.float32, device=self.hps['device']) * c_noise + 1)

        for i in range(n_rand):
            z = torch.randn((n_rand, self.hps['feat_in']), dtype=torch.float32, device=self.hps['device'])
            in_pred[i, :, :] = self.reverse(
                torch.randn((n, self.hps['feat_in']), dtype=torch.float32, device=self.hps['device']), c=c).to('cpu')

        return torch.std_mean(in_pred, dim=0)

    def getParList(self):
        self.trainable_parameters = nn.ParameterList()
        for p in self.cinn.parameters():
            if p.requires_grad:
                self.trainable_parameters.append(p)

        for p in self.condnet.parameters():
            if p.requires_grad:
                self.trainable_parameters.append(p)

    def __sub_fc(self, c_in, c_out, idx):
        modules = [nn.Linear(c_in, self.hps['subnet_dim'][idx])]

        for i in range(self.hps['subnet_depth'][idx] - 1):
            if self.hps['subnet_dropout']:
                modules.append(nn.Dropout(p=self.hps['subnet_dropout_rate']))
            if self.hps['subnet_batch_norm']:
                modules.append(nn.BatchNorm1d(self.hps['subnet_dim'][idx]))
            modules.append(self.hps['subnet_activation'])
            modules.append(nn.Linear(self.hps['subnet_dim'][idx], self.hps['subnet_dim'][idx]))

        if self.hps['subnet_dropout']:
            modules.append(nn.Dropout(p=self.hps['subnet_dropout_rate']))
        if self.hps['subnet_batch_norm']:
            modules.append(nn.BatchNorm1d(self.hps['subnet_dim'][idx]))

        modules.append(self.hps['subnet_activation'])
        modules.append(nn.Linear(self.hps['subnet_dim'][idx], c_out))

        return nn.Sequential(*modules)

    def __getDataLoad(self, inputs, conditions, setscaler=False):

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float, device=self.hps['device'])
        else:
            inputs = inputs.clone()
            inputs = inputs.to(device=self.hps['device'])

        if not torch.is_tensor(conditions):
            conditions = torch.tensor(conditions, dtype=torch.float, device=self.hps['device'])
        else:
            conditions = conditions.clone()
            conditions = conditions.to(device=self.hps['device'])

        inputs = self.in_scaler.transform(inputs, set_par=setscaler)

        conditions = self.cond_scaler.transform(conditions, set_par=setscaler)

        t_idx = int(inputs.shape[0] * (1 - self.hps['test_split']))

        if conditions.ndim == 2:
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs[:t_idx, :], conditions[:t_idx, :]),
                batch_size=self.hps['batch_size'], shuffle=True, drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs[t_idx:, :], conditions[t_idx:, :]),
                batch_size=inputs.shape[0]-t_idx, shuffle=True, drop_last=True)

        elif conditions.ndim == 3:
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs[:t_idx, :], conditions[:t_idx, :, :]),
                batch_size=self.hps['batch_size'], shuffle=True, drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs[t_idx:, :], conditions[t_idx:, :, :]),
                batch_size=inputs.shape[0]-t_idx, shuffle=True, drop_last=True)

        return train_loader, val_loader

    @staticmethod
    def default_hps():
        """ Return default hyper-parameters """
        hps_dict = {

            # Model Size:
            'feat_in': None,
            'feat_cond': None,

            # Experiment Params:
            'n_CCB': 10,  # number of coupling block
            'permute_soft': False,  # permutation soft of coupling block
            'subnet_depth': 1,  # number of dense hidden layer in affine coupling block subnet
            'subnet_dim': 300,  # perceptron numbers per hidden layer in affine coupling block subnet
            'subnet_activation': nn.PReLU(),  # activation function of each hidden layer in affine coupling block subnet
            'subnet_batch_norm': False,
            'subnet_dropout': True,
            'subnet_dropout_rate': 0.125,

            'condnet_cluster': 3,
            'condnet_inter_dim_pow': 7,  # output dimension of condition net
            'condnet_out_dim': 200,  # activation function for condition net
            'condnet_activation': nn.PReLU(),  # activation function for condition net
            'condnet_pooling': True,  # activation function for condition net

            # Training Params:
            'clamp': 2.0,  # necessary input for Couplinblock
            'optimizer': 'adagrad',  # optimizer
            'learning_rate': 0.075,  # learning rate optimizer
            'optim_eps': 1e-6,
            'weight_decay': 2e-5,
            'lr_scheduler': False,
            'lr_epoch_step': 10,
            'lr_factor': 0.75,
            'c_noise': None,
            'epochs': 50,
            'batch_size': 16,  # batch size
            'test_split': 0.1,
            'grad_clip': 0.01,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # device name
        }

        return hps_dict

    def getOptimizer(self):

        if self.hps['optimizer'].lower() == 'adadelta':
            return torch.optim.Adadelta(self.trainable_parameters, lr=self.hps['learning_rate'])  # ,
            # eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])
        elif self.hps['optimizer'].lower() == 'adagrad':
            return torch.optim.Adagrad(self.trainable_parameters, lr=self.hps['learning_rate'])  # ,
            # eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])
        elif self.hps['optimizer'].lower() == 'adamw':
            return torch.optim.AdamW(self.trainable_parameters, lr=self.hps['learning_rate'], betas=(0.8, 0.9))  # ,
            # eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'], amsgrad=False)
        elif self.hps['optimizer'].lower() == 'adamax':
            return torch.optim.Adamax(self.trainable_parameters, lr=self.hps['learning_rate'], betas=(0.8, 0.9))  # ,
            # eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])
        else:
            return torch.optim.Adam(self.trainable_parameters, lr=self.hps['learning_rate'], betas=(0.8, 0.9))  # ,
            # eps=self.hps['optim_eps'], weight_decay=self.hps['weight_decay'])

    def init_dicts(self):
        print('initdicts notused')

    def train_model(self, inputs, conditions, verbose=True, tune_report=False, resume=False,
                    check_int=None, check_dir='./ChkPnt', save_file=None):
        """training method of the model"""

        def print_train_progress(model, verbose, state, time_step=None, epoch=None):
            if not verbose:
                return
            if state == 'hps':
                print('\n\n The Models hyperparameter set is:')
                for key, value in self.hps.items():
                    print(key, ' : ', value)
            elif state == 'start':
                print('\n\n| Epoch:    |  Time:  | l-rate: | Loss_maxL_train: |  | Acc_i_val: | L_maxL_val: |')
                print('|-----------|---------|---------|------------------|  |------------|-------------|')
            elif state == 'progress':
                print('| %4d/%4d | %6ds | %6.5f | %16.5f |  | %10.6f | %11.5f |' % (
                    epoch + 1, self.hps['epochs'], min((time() - time_step), 99999),
                    self.optimizer.param_groups[0]['lr'],
                    min(self.train_loss[epoch], 9999),
                    min(np.nanmean(self.val_acc[epoch], axis=0), 1.),
                    min(self.val_loss[epoch], 9999)))
            elif state == 'end':
                print('|-----------|---------|---------|------------------|  |------------|-------------|')
                print(f"\n\nTraining took {(time() - time_step) / 60:.2f} minutes\n")

        # Get Data Loaders
        train_loader, val_loader = self.__getDataLoad(inputs, conditions, setscaler=True)

        # Initialize Output
        t_start = time()
        print_train_progress(self, verbose, 'hps')
        print_train_progress(self, verbose, 'start')

        # Check if training should be resumed
        if resume:
            epoch_resume = np.where(np.array(self.val_loss) != None)
            epoch_resume = int(epoch_resume[0][-1] + 1)
            self.train_loss.extend([None] * (self.hps['epochs'] - epoch_resume))
            self.val_acc.extend([None] * (self.hps['epochs'] - epoch_resume))
            self.val_loss.extend([None] * (self.hps['epochs'] - epoch_resume))

        else:
            # Init Dicts
            epoch_resume = 0
            self.train_loss = [None] * self.hps['epochs']
            self.val_acc = [None] * self.hps['epochs']
            self.val_loss = [None] * self.hps['epochs']

        # Initialize learining rate scheduler
        if self.hps['lr_scheduler']:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hps['lr_epoch_step'],
                                                           gamma=self.hps['lr_factor'])

        # Save hps and data to log file if check int is defined
        if check_int:
            with open(check_dir + 'model_config.txt', 'w') as f:
                f.write('The cINN configuration is as follows:\n\n')
                f.write('Samples sizes: %d\n\n' % (inputs.shape[0]))
                f.write('The Models hyperparameter set is:\n')
                for key, value in self.hps.items():
                    f.write(str(key) + ' : ' + str(value) + '\n')
                f.write('\nThe Input channel names are:\n')
                for ch in self.ch_name:
                    f.write(ch + '\n')

        # Start Training
        for i_epoch in range(epoch_resume, self.hps['epochs']):
            t_epoch = time()

            # Train Model
            self.__train_process(train_loader, i_epoch=i_epoch)

            # Validate Model
            self.__val_process(val_loader, i_epoch=i_epoch)

            # Output epoch step
            print_train_progress(self, verbose, 'progress', time_step=t_epoch, epoch=i_epoch)

            # Step learining rate scheduler
            if self.hps['lr_scheduler']:
                lr_scheduler.step()

            # Report to hps-tuning
            if tune_report:
                if math.isnan(self.train_loss[i_epoch]):
                    done_flag = True
                else:
                    done_flag = False

                tune.report(done=done_flag,
                            L_maxL_train=self.train_loss[i_epoch],
                            Acc_in_val=np.nanmean(self.val_acc[i_epoch], axis=0),
                            L_maxL_val=self.val_loss[i_epoch])

            # Save Checkpoint
            if check_int and (i_epoch + 1) % check_int == 0:
                try:
                    os.rmdir(check_dir)
                except:
                    pass

                try:
                    os.mkdir(check_dir)
                except:
                    pass

                fname = check_dir + '/ChkPnt_Ep_' + str(i_epoch + 1) + '.pt'
                self.save_state(fname)
                fname = check_dir + '/TrainingProgress_Ep_' + str(i_epoch + 1) + '.png'
                self.print_training(fname)

        # Print End
        print_train_progress(self, verbose, 'end', time_step=t_start)

        if save_file:
            self.save_state(save_file)

    def __train_process(self, train_loader, i_epoch=0):
        self.cinn.train()
        self.condnet.train()
        loss_history = []

        for x, c in train_loader:
            x, c = x.to(self.hps['device']), c.to(self.hps['device'])

            self.optimizer.zero_grad()

            if self.hps['c_noise']:
                c = c * (torch.randn_like(c, dtype=torch.float32, device=self.hps['device']) * self.hps['c_noise'] + 1)

            # Forward step:
            z, log_jac_det = self.forward(x, c)

            # maxlikelyhood loss:
            l_maxL = self.__max_Likelyhood(z, log_jac_det)

            l_maxL.backward()

            loss_history.append(l_maxL.item())

            nn.utils.clip_grad_norm_(self.trainable_parameters, self.hps['grad_clip'])

            self.optimizer.step()

        epoch_losses = np.mean(np.array(loss_history), axis=0)

        self.train_loss[i_epoch] = epoch_losses

    def __val_process(self, val_loader, i_epoch=0):

        self.cinn.eval()
        self.condnet.eval()
        with torch.no_grad():

            loss_history = []
            acc_in = []
            max_it = 150
            it = 0
            for x, c in val_loader:
                it += 1
                if it > max_it:
                    break

                x, c = x.to(self.hps['device']), c.to(self.hps['device'])

                # Forward step:
                z, log_jac_det = self.forward(x, c)

                # maxlikelyhood loss:
                l_maxL = self.__max_Likelyhood(z, log_jac_det)

                # Backward step:
                z = torch.randn_like(z, dtype=torch.float, device=self.hps['device'])
                x_rec = self.reverse(z, c)

                acc_in.append(self.__R2(x, x_rec))

                loss_history.append(l_maxL.item())

            epoch_losses = np.mean(np.array(loss_history), axis=0)

            self.val_loss[i_epoch] = epoch_losses
            self.val_acc[i_epoch] = np.nanmean(np.stack(acc_in, axis=0), axis=0)

    @staticmethod
    def hps_tune(inputs, conditions, hps=None, epochs=50, config=None, n_samples=10, cpus_per_trial=5,
                 gpus_per_trial=0.5,
                 max_n_epochs=None):

        if not max_n_epochs or max_n_epochs > epochs:
            scheduler = ASHAScheduler(metric="L_maxL_val", mode="min")
        else:
            scheduler = ASHAScheduler(metric="L_maxL_val", mode="min", grace_period=max_n_epochs)

        if not hps:
            hps = FullModel.default_hps()

        if not config:
            config = {
                'n_CCB': hp.quniform('n_CouplingBlock', 4, 60, 4),
                'subnet_layer': hp.quniform('subnet_layer', 1, 10, 1),
                'subnet_dim': hp.quniform('subnet_dim', 20, 200, 5),
                'subnet_activation': hp.choice('activation',
                                               [nn.ReLU(), nn.LeakyReLU(), nn.PReLU(), nn.Tanhshrink(), nn.Tanh(),
                                                nn.CELU()]),
                'optimizer': hp.choice('optimizer', ['adamax', 'adadelta', 'adagrad', 'adamw', 'adam']),
                'learning_rate': hp.quniform('learning_rate', 0.01, 0.05, 0.005),
                'batch_size': hp.choice('batch_size', [128, 256, 512])
            }

        def train_tuning(config):

            # Defines how to cast values default ist float
            integers = ['n_CCB', 'subnet_depth', 'subnet_dim', 'condnet_cluster',
                        'condnet_inter_dim_pow', 'condnet_out_dim', 'lr_epoch_step', 'batch_size']

            # Define tuning
            for key in config:
                if key in integers:
                    hps[key] = int(config[key])
                else:
                    hps[key] = config[key]

            hps['epochs'] = epochs
            hps['feat_in'] = inputs.shape[1]
            hps['feat_cond'] = (conditions.shape[2], conditions.shape[1])
            hps['condnet_out_dim'] = conditions.shape[1] * conditions.shape[2]
            n = np.ceil(hps['n_CCB'] / hps['condnet_cluster'])
            hps['condnet_out_dim'] = np.cumsum(np.ones(int(n), dtype=int) * 100) + int(
                hps['condnet_out_dim'] - 100 * n)  # Out.shape[1]*Out.shape[2]
            hps['condnet_out_dim'][hps['condnet_out_dim'] < 100] = 100

            model = FullModel(hps=hps)

            model.train_model(inputs, conditions, verbose=False, tune_report=True)

        # Defining search algorithm
        search_algo = HyperOptSearch(space=config, metric="L_maxL_val", mode="min")

        result = tune.run(
            train_tuning,
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            num_samples=n_samples,
            scheduler=scheduler,
            search_alg=search_algo)

        best_trial = result.get_best_trial("L_maxL_val", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["L_maxL_val"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["Acc_in_val"]))

        return result

    @staticmethod
    def __max_Likelyhood(z, log_jac_det):
        return torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_jac_det)

    @staticmethod
    def __R2(inputs, targets):
        R2 = [np.corrcoef(inputs[:, i].cpu().detach().numpy(), targets[:, i].cpu().detach().numpy())[0, 1] ** 2 for i in
              range(inputs.shape[1])]
        return np.array(R2)

    def update_hps(self, hps=None):
        """ method to update the hyper parameter set"""
        if hps:
            for k, v in hps.items():
                self.hps[k] = v

        self.__init__(hps=self.hps)

    def save_state(self, fname):
        torch.save({'cinn_state': self.cinn.state_dict(),
                    'condnet_state': self.condnet.state_dict(),
                    'hps': self.hps,
                    'opt': self.optimizer.state_dict(),
                    'in_scaler': self.in_scaler.save(),
                    'cond_scaler': self.cond_scaler.save(),
                    'ch_name_tex': self.ch_name_tex,
                    'ch_name': self.ch_name,
                    'train_loss': self.train_loss,
                    'val_loss': self.val_loss,
                    'val_acc': self.val_acc}, fname)

    def load_state(self, fname):
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = torch.load(fname, map_location=dev)
        data['cinn_state'] = {k: v for k, v in data['cinn_state'].items() if 'tmp_var' not in k}
        data['condnet_state'] = {k: v for k, v in data['condnet_state'].items() if 'tmp_var' not in k}

        data['hps']['device'] = dev

        self.update_hps(data['hps'])

        self.cinn.load_state_dict(data['cinn_state'])
        self.condnet.load_state_dict(data['condnet_state'])

        self.in_scaler.load(data['in_scaler'])
        self.cond_scaler.load(data['cond_scaler'])

        self.optimizer.load_state_dict(data['opt'])

        self.train_loss = data['train_loss']
        self.val_loss = data['val_loss']
        self.val_acc = data['val_acc']
        if 'ch_name' in data.keys():
            self.ch_name = data['ch_name']
        if 'ch_name_tex' in data.keys():
            self.ch_name_tex = data['ch_name_tex']

        self.getParList()

    @staticmethod
    def loadmodel(fname, verbose=False):
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = torch.load(fname, map_location=dev)

        data['hps']['device'] = dev

        hps = FullModel.default_hps()

        for k, v in data['hps'].items():
            if k in hps.keys():
                hps[k] = v

        model = FullModel(hps=hps, verbose=verbose)

        model.load_state(fname)

        return model

    def print_training(self, fname='Training.png', dpi=150, figsize=[13, 5]):
        plt.rc('text', usetex=True)
        fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        L_maxL_val = np.array(self.val_loss)

        L_maxL_train = np.array(self.train_loss)

        n = np.where(L_maxL_train != None)
        n = n[0][-1] + 1

        val_acc = np.vstack(self.val_acc[:n])
        val_acc[np.isnan(val_acc)] = 0
        ep = np.arange(0, n, 1)
        val_sel = np.arange(0,n,2)
        axs[0].plot(ep[val_sel] + 1, L_maxL_val[val_sel], 'r', label='$\mathcal{L}_\mathrm{NLL,val}$')
        axs[0].plot(ep + 1, L_maxL_train[:n], ':', color=(0, 0.4, 0.58), label='$\mathcal{L}_\mathrm{NLL,train}$')
        axs[0].set_ylim([min([np.amin(L_maxL_train[:n]), np.amin(L_maxL_val[val_sel]), 0]), 100])
        axs[0].set_ylabel('$\mathcal{L}_\mathrm{NLL}$', fontsize='small')
        axs[0].set_xlabel('Epochs', fontsize='small')
        axs[0].set_xlim(0, n)
        axs[0].grid(b=True, which='major', axis='both')
        axs[0].legend(loc='upper right', fontsize='small')

        feat = val_acc.shape[-1]
        cm = plt.get_cmap('gist_rainbow')
        cNorm = colors.Normalize(vmin=0, vmax=feat - 1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        axs[1].set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(feat)])

        for i in range(feat):
            axs[1].plot(ep[val_sel] + 1, val_acc[val_sel, i], label=self.ch_name_tex[i])
            axs[1].text(ep[val_sel[-1]] + ep[val_sel[-1]] / 50, val_acc[val_sel[-1], i], self.ch_name_tex[i], va='center', fontsize='xx-small',
                        usetex=True)

        axs[1].set_title('mean accuracy= %6.5f' % (np.mean(val_acc[n - 1, :])), fontsize='small')
        axs[1].set_ylabel('Accuracy', fontsize='small')
        axs[1].set_xlabel('Epochs', fontsize='small')
        axs[1].set_ylim([np.nanmin(val_acc), np.nanmax(val_acc)])
        axs[1].set_xlim(0, n)
        axs[1].grid(b=True, which='major', axis='both')
        axs[1].legend(loc=(1.4, 0), fontsize='xx-small', ncol=2)

        axs[2].axis('off')
        plt.savefig(fname)
        plt.close('all')
