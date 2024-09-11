
from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional
from model import Model
from DCLS.construct.modules import  Dcls3_1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class SnnDelays(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

    # Try factoring this method
    # Check ThresholdDependent batchnorm (in spikingjelly)
    def build_model(self):

        ########################### Model Description :
        #
        #  self.blocks = (n_layers,  0:weights+bn  |  1: lif+dropout+(synapseFilter) ,  element in sub-block)
        #


        ################################################   First Layer    #######################################################
        
        self.blocks = [[[Dcls3_1d(in_channels = self.config.n_inputs, out_channels = self.config.n_hidden_neurons,
                                   kernel_count=self.config.kernel_count, groups = 1,
                                   dilated_kernel_size = self.config.max_delay,
                                   bias=self.config.bias,
                                   dense_kernel_size=(3, 3), # no learnable positions in these 2 dims
                                   padding=(1, 1, self.config.max_delay // 2),
                                   version=self.config.DCLSversion)],

                        [layer.Dropout(self.config.dropout_p, step_mode='m')],
                        [layer.MaxPool2d(2, 2, step_mode='m')]]]

        self.blocks[0][0].insert(1, layer.BatchNorm2d(self.config.n_hidden_neurons, step_mode='m'))
        self.blocks[0][1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset,
                                                       step_mode='m', decay_input=False, store_v_seq = True))


        ################################################   Hidden Layers    #######################################################

        for i in range(1,self.config.n_hidden_layers): # i =  1, 2, 3, 4
            self.block = [[Dcls3_1d(in_channels = self.config.n_hidden_neurons,  out_channels = self.config.n_hidden_neurons,
                                   kernel_count=self.config.kernel_count, groups = 1,
                                   dilated_kernel_size = self.config.max_delay,
                                   bias=self.config.bias,
                                   dense_kernel_size=(3, 3), # no learnable positions in these 2 dims
                                   padding=(1, 1, self.config.max_delay // 2),
                                   version=self.config.DCLSversion)],

                            [layer.Dropout(self.config.dropout_p, step_mode='m')],
                            [layer.MaxPool2d(2, 2, step_mode='m')]]

            self.block[0].insert(1, layer.BatchNorm2d(self.config.n_hidden_neurons, step_mode='m'))
            self.block[1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset,
                                                       step_mode='m', decay_input=False, store_v_seq = True))

            self.blocks.append(self.block)

        

        ################################################   Final Layer    #######################################################

        
        final_width = self.config.width//(2**self.config.n_hidden_layers) # the width of the last layer (<128 because of the pooling)
        dim_flatten = self.config.n_hidden_neurons * final_width * final_width # the dimension to flatten (channels, height, width) to 

        print(dim_flatten)

        self.final_block  = nn.Sequential(
            layer.Flatten(step_mode='m'),
            layer.Dropout(0.5, step_mode='m'),
            layer.Linear(dim_flatten, 512, step_mode='m'),
            neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset,
                                                       step_mode='m', decay_input=False, store_v_seq = True),

            layer.Dropout(0.5,step_mode='m'),
            layer.Linear(512, 110,step_mode='m'),
            neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset,
                                                       step_mode='m', decay_input=False, store_v_seq = True)
        )
        self.voting_layer = layer.VotingLayer(10, step_mode = 'm')

        self.model = [l for block in self.blocks for sub_block in block for l in sub_block] + [self.final_block] + [self.voting_layer]
        self.model = nn.Sequential(*self.model)
        print(self.model)

        self.positions = []
        self.weights = []
        self.weights_bn = []
        self.weights_plif = []

        for m in self.model.modules():
            if isinstance(m, Dcls3_1d):
                self.positions.append(m.P)
                self.weights.append(m.weight)
                if self.config.bias:
                    self.weights_bn.append(m.bias)
            elif isinstance(m, layer.Linear):
                self.weights.append(m.weight)
            elif isinstance(m, layer.BatchNorm2d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)


    def init_model(self):
        self.mask = []

        for i in range(self.config.n_hidden_layers):
                # can you replace with self.positions?
                torch.nn.init.uniform_(self.blocks[i][0][0].P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0][0].clamp_parameters()


        for i in range(self.config.n_hidden_layers):
            # can you replace with self.positions?
            torch.nn.init.constant_(self.blocks[i][0][0].SIG, self.config.sigInit)
            self.blocks[i][0][0].SIG.requires_grad = False




    def reset_model(self, train=True):
        functional.reset_net(self)

        for i in range(self.config.n_hidden_layers):
            if self.config.sparsity_p > 0:
                with torch.no_grad():
                    self.mask[i] = self.mask[i].to(self.blocks[i][0][0].weight.device)
                    #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                    self.blocks[i][0][0].weight *= self.mask[i]

        # We use clamp_parameters of the Dcls1d modules
        if train:
            for block in self.blocks:
                block[0][0].clamp_parameters()




    def decrease_sig(self, epoch):

        # Decreasing to 0.23 instead of 0.5

        alpha = 0
        sig = self.blocks[-1][0][0].SIG[0,0,0,0].mean((0,1)).detach().cpu().item()
        if self.config.decrease_sig_method == 'exp':
            if epoch < self.config.final_epoch and sig > 0.23:
                alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for block in self.blocks:
                    block[0][0].SIG *= alpha
                    # No need to clamp after modifying sigma
                    #block[0][0].clamp_parameters()


    def forward(self, x):

        for block_id in range(self.config.n_hidden_layers):
            # x is permuted: (time, batch, c,x,y) => (batch,  c,x,y, time)  in order to be processed by the convolution
            x = x.permute(1,2,3,4,0)
            x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)  # we use padding for the delays kernel

            # we use convolution of delay kernels
            x = self.blocks[block_id][0][0](x)
            # We permute again: (batch,  c,x,y, time) => (time, batch,  c,x,y) in order to be processed by batchnorm or Lif
            x = x.permute(4,0,1,2,3)

            x = self.blocks[block_id][0][1](x)
            # we use our spiking neuron filter
            spikes = self.blocks[block_id][1][0](x)
            # we use dropout on generated spikes tensor
            x = self.blocks[block_id][1][1](spikes)
            
            #Pooling
            x = self.blocks[block_id][2][0](x)

            # x is back to shape (time, batch, neurons)

        # Finally, we apply same transforms for the output layer
        #   (time, batch, neurons) => (batch, neurons, time)
        x = x.permute(1,2,3,4,0)
        x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)
        
        # permute : (batch, neurons, time) => (time, batch, neurons)  For final spiking neuron filter
        x = x.permute(4,0,1,2,3)
        # Apply final layer
        x = self.final_block(x)
        x = self.voting_layer(x)
        return x


    def round_pos(self):
        with torch.no_grad():
            for i in range(len(self.blocks)):
                self.blocks[i][0][0].P.round_()
                self.blocks[i][0][0].clamp_parameters()