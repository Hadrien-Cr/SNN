from spikingjelly.activation_based import surrogate

class Config:

    ################################################
    #            General configuration             #
    ################################################

    seed = 0
    time_step = 16
    batch_size = 4
    n_hidden_layers = 5
    n_hidden_neurons = 128
    n_inputs = 2
    n_outputs = 11

    time_step = 16
    learn_delay = True

    lr_w = 1e-3
    lr_pos = 100*lr_w
    weight_decay = 1e-5
    dropout_p = 0.25

    height = 128
    width = 128

    ################################################
    #               Model Achitecture              #
    ################################################
    init_tau = 2*time_step # in ms, can't be < time_step
                     
    use_maxpool = True

    use_batchnorm = True
    bias = True
    detach_reset = True

    output_v_threshold =  1e9  

    v_threshold = 1.0
    alpha = 2.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)

    init_w_method = 'kaiming_uniform'

    init_tau = (init_tau  +  1e-9) / time_step


    ################################################
    #                    Delays                    #
    ################################################

    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 128//time_step

    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number

    sigInit = max_delay // 2         

    DCLSversion =  'gauss'
    left_padding = max_delay-1
    right_padding = (max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2


config_snn_delays = Config()