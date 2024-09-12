from spikingjelly.activation_based import surrogate

class Config:

    ################################################
    #            General configuration             #
    ################################################

    seed = 0
    time_step = 16
    batch_size = 4
    n_inputs = 2
    n_hidden_layers = 5
    n_hidden_neurons = 128
    n_outputs = 11

    lr_w = 1e-3
    weight_decay = 1e-5
    dropout_p = 0.25


config_snn_no_delays = Config()