from spikingjelly.activation_based import surrogate

class Config:

    ################################################
    #            General configuration             #
    ################################################

    seed = 0

    # model type could be set to : 'snn_delays' |  'snn_delays_lr0' |  'snn'
    model_type = 'snn_delays'
    save_model_path = 'snn_delays_REPL.pt'
    time_step = 16

    n_bins = 5

    epochs = 50
    batch_size = 4

    ################################################
    #               Model Achitecture              #
    ################################################
    spiking_neuron_type = 'lif'         # plif, lif
    init_tau = 20                   # in ms, can't be < time_step

    stateful_synapse_tau = 20      # in ms, can't be < time_step
    stateful_synapse = False
    stateful_synapse_learnable = False

    n_inputs = 2

    height = 128
    width = 128

    n_hidden_layers = 5
    n_hidden_neurons = 128

    n_outputs = 11

    sparsity_p = 0

    dropout_p = 0.25
    use_batchnorm = True
    bias = False
    detach_reset = True


    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum
    loss_fn = 'CEloss'
    output_v_threshold = 2.0 if loss == 'spike_count' else 1e9  #use 1e9 for loss = 'mean' or 'max'

    v_threshold = 1.0
    alpha = 5.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)

    init_w_method = 'kaiming_uniform'

    init_tau = (init_tau  +  1e-9) / time_step
    stateful_synapse_tau = (stateful_synapse_tau  +  1e-9) / time_step


    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    weight_decay = 1e-5

    lr_w = 1e-3
    lr_pos = 100*lr_w 

    # 'one_cycle', 'cosine_a', 'none'
    scheduler_w = 'one_cycle'
    scheduler_pos = 'cosine_a' 


    # for one cycle
    max_lr_w = 5 * lr_w
    max_lr_pos = 5 * lr_pos


    # for cosine annealing
    t_max_w = epochs
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 64//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number

    sigInit = max_delay // 2       
    final_epoch = (1*epochs)//4     

    DCLSversion =  'gauss'
    left_padding = max_delay-1
    right_padding = (max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2


    use_wandb = False


config = Config()