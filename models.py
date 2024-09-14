from spikingjelly.activation_based import neuron, layer, surrogate
from spikingjelly.activation_based import functional
from DCLS.construct.modules import  Dcls3_1d, Dcls1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

class Net_No_Delays(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        conv = []
        for i in range(config.n_hidden_layers):
            if conv.__len__() == 0:
                in_channels = config.n_inputs
            else:
                in_channels = config.n_hidden_neurons

            conv.append(layer.Conv2d(in_channels, config.n_hidden_neurons, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(config.n_hidden_neurons))
            conv.append(neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                        surrogate_function=surrogate.ATan(), detach_reset=self.config.detach_reset,
                                                        step_mode='m'))
            conv.append(layer.Dropout(config.dropout_p))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_cat = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(config.n_hidden_neurons * 4 * 4, 512),
            neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                        surrogate_function=surrogate.ATan(), detach_reset=self.config.detach_reset,
                                                        step_mode='m'),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                        surrogate_function=surrogate.ATan(), detach_reset=self.config.detach_reset,
                                                        step_mode='m'),  
        )
        self.voting_layer = layer.VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = self.conv_cat(x)
        x = self.fc(x)
        x = self.voting_layer(x)
        return x 
    
    def reset_model(self, train=True):
        functional.reset_net(self)


class Dcls3_1d_SJ(Dcls3_1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_count,
        learn_delay=True,
        stride=1,
        spatial_padding=0,
        dense_kernel_size=1,
        dilated_kernel_size=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        version='gauss',
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_count,
            (*stride, 1),
            (*spatial_padding, 0),
            dense_kernel_size,
            dilated_kernel_size,
            groups,
            bias,
            padding_mode,  
            version,
        )

        self.learn_delay = learn_delay

        if not self.learn_delay:
            self.P.requires_grad = False

        if self.version == 'gauss':
            self.SIG.requires_grad = False
            self.sig_init = self.dilated_kernel_size[0]/2
            torch.nn.init.constant_(self.SIG, self.sig_init)

    def decrease_sig(self, epoch, epochs):
        if self.version == 'gauss':
            final_epoch = (1*epochs)//4
            final_sig = 0.23
            sig = self.SIG[0, 0, 0, 0, 0, 0].detach().cpu().item()
            alpha = (final_sig/self.sig_init)**(1/final_epoch)
            if epoch < final_epoch and sig > final_sig:
                self.SIG *= alpha

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # [N, T, C, H, W] -> [N, C, H, W, T]
        x = F.pad(x, (self.dilated_kernel_size[0]-1, 0), mode='constant', value=0)
        x = super().forward(x)
        x = x.permute(0, 4, 1, 2, 3) # [N, C, H, W, T] -> [N, T, C, H, W]
        return x
    

class Net_With_Delays(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ############# Model Description #############
        # a:  DCLS_Block (repeated n_hidden_layers times)
        # b: Fully Connected Block (without learning delays, using Flatten and layer.Linear)
        # c: Voting Layer
        
        ################################################
        #  a DCLS_Block is made of : 
        #    0: (DCLS_3_1d + batchnorm) | 
        #    1: (lif+dropout+(synapseFilter)) , 
        #    2: (pooling) |


        #####################  a. DCLS_Blocks   #########
        
        self.blocks = []
        
        for i in range(0,self.config.n_hidden_layers): # i =  0,1, 2, 3, 4
            if i == 0:
                in_channels = self.config.n_inputs
            else:
                in_channels = self.config.n_hidden_neurons

            self.block = [[Dcls3_1d_SJ(in_channels = in_channels, 
                                    out_channels = self.config.n_hidden_neurons,
                                    kernel_count=self.config.kernel_count, 
                                    groups = 1,
                                    learn_delay = self.config.learn_delay,
                                    dilated_kernel_size = self.config.max_delay,
                                    spatial_padding = (1,1),
                                    stride = (1,1),
                                    dense_kernel_size=(3,3), 
                                    bias=self.config.bias,
                                    version=self.config.DCLSversion
                                    ),
                            layer.BatchNorm2d(self.config.n_hidden_neurons, step_mode='m')],

                            [neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                        surrogate_function=surrogate.ATan(), detach_reset=self.config.detach_reset,
                                                        step_mode='m'),
                            layer.Dropout(self.config.dropout_p, step_mode='m'),]
                        ]

            if self.config.use_maxpool:
                self.block.append([layer.MaxPool2d(2, 2, step_mode='m')])

            self.blocks.append(self.block)

        

        ############  b. Final BLOCK (Fully conected)   #########################

        final_width = self.config.width//(2**self.config.n_hidden_layers) # the width of the last image (<128 because of the max pooling)
        dim_flatten = self.config.n_hidden_neurons * final_width * final_width # the dimension after flattening the last imge from (time,batch_size,channels, height, width) to (time, batch_size, *)

        self.final_block  = [
            [layer.Flatten(step_mode='m')],

            [layer.Dropout(0.5, step_mode='m'),
            layer.Linear(dim_flatten, 512, step_mode='m'),
            neuron.LIFNode(tau=self.config.init_tau, 
                            v_threshold=self.config.v_threshold,
                            surrogate_function=surrogate.ATan(), 
                            detach_reset=self.config.detach_reset,
                            step_mode='m')],

            [layer.Dropout(0.5,step_mode='m'),
            layer.Linear(512, 110,step_mode='m'),
            neuron.LIFNode(tau=self.config.init_tau, 
                            v_threshold=self.config.v_threshold,
                            surrogate_function=surrogate.ATan(), 
                            detach_reset=self.config.detach_reset,
                            step_mode='m')]
            ]
        

        ############  c. Voting Layer  #####################
        self.voting_layer = layer.VotingLayer(10, step_mode = 'm')


        ############ Adding all up  #########################
        self.model = [l for block in self.blocks for sub_block in block for l in sub_block] + [l for sub_block in self.final_block for l in sub_block] + [self.voting_layer]
        self.model = nn.Sequential(*self.model)
        print(self.model)

        self.positions = []
        self.weights = []
        self.weights_bn = []


        # handle parameters asignments
        for m in self.model.modules():

            if isinstance(m, Dcls3_1d_SJ):
                self.positions.append(m.P)
                self.weights.append(m.weight)
                if self.config.bias:
                    self.weights_bn.append(m.bias)

            elif isinstance(m, layer.Linear):
                self.weights.append(m.weight)
                self.weights.append(m.bias)

            elif isinstance(m, layer.BatchNorm2d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)

        self.init_model()

        self.delays_histogram = []
        self.delays_histogram_first_delay = []

    def init_model(self):

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers):
                # can you replace with self.weights ?
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].weight, nonlinearity='relu')

        for i in range(self.config.n_hidden_layers):

            if self.config.init_pos_method == 'right':
                torch.nn.init.constant_(self.blocks[i][0][0].P, (self.config.max_delay[0] // 2))
                self.blocks[i][0][0].clamp_parameters()

            elif self.config.init_pos_method == 'middle':
                torch.nn.init.constant_(self.blocks[i][0][0].P, 0.0)
                self.blocks[i][0][0].clamp_parameters()

            elif self.config.init_pos_method == 'uniform':
                torch.nn.init.uniform_(self.blocks[i][0][0].P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0][0].clamp_parameters()


    def reset_model(self, train=True):
        functional.reset_net(self)
        if train: 
            for block in self.blocks:
                block[0][0].clamp_parameters()

    def decrease_sig(self, epoch, epochs):
        for i in range(self.config.n_hidden_layers):
            self.blocks[i][0][0].decrease_sig(epoch,epochs)

    def forward(self, x):

        for block_id in range(self.config.n_hidden_layers):
            # feed DCLS
            x = self.blocks[block_id][0][0](x)
            
            # # feed (BatchNorm)
            x = self.blocks[block_id][0][1](x)

            # Feed the dropout layer
            x = self.blocks[block_id][1][0](x)

            # Feed the LIF
            x = self.blocks[block_id][1][1](x)
            
            # Feed the Pooling
            if self.config.use_maxpool:
                x = self.blocks[block_id][2][0](x)

            # x is back to shape (time, batch, neurons)

        # Flatten
        x = self.final_block[0][0](x)

        # 1st Linear layer
        x = self.final_block[1][0](x) #drop out
        x = self.final_block[1][1](x) #linear
        x = self.final_block[1][2](x) #neuron

        # 2nd Linear layer  
        x = self.final_block[2][0](x) #drop out
        x = self.final_block[2][1](x) #linear
        x = self.final_block[2][2](x) #neuron

        output  = self.voting_layer(x)

        return output


    def round_pos(self):
        with torch.no_grad():
            for i in range(len(self.blocks)):
                self.blocks[i][0][0].P.round_()
                self.blocks[i][0][0].clamp_parameters()


    def collect_delays(self):
        with torch.no_grad():
            delays = torch.concat([p.flatten() for p in self.positions]).flatten()

        max_delay = self.config.max_delay

        values_per_bins = [0 for delay in range(-max_delay//2, max_delay//2+1)]

        for delay in delays:
            bin = round(delay.item())
            values_per_bins[bin]+=1

        self.delays_histogram.append(values_per_bins)
        
        first_delay = delays[0].item()
        values_per_bins_first_delay = [np.exp(-(delay-first_delay)**2/(self.blocks[0][0][0].SIG if self.config.DCLSversion == "gauss" else 1e-7)) for delay in range(-max_delay//2, max_delay//2+1)]
        self.delays_histogram_first_delay.append(values_per_bins_first_delay)


    def draw_delays_all_evolution(self):
        fig,ax = plt.subplots()
        im = ax.imshow(self.delays_histogram)
        fig.tight_layout()
        plt.savefig('delays_all_evolution.png')

    def draw_delays_single_evolution(self):
        fig,ax = plt.subplots()
        im = ax.imshow(self.delays_histogram_first_delay)
        fig.tight_layout()
        plt.savefig('delays_single_evolution.png')