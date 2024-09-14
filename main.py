import torch
import sys
import torch.nn.functional as F
from torch import optim
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import models

from torch.utils.tensorboard import SummaryWriter
import time,os,argparse,datetime,yaml
from tqdm import tqdm

def  print_model_size(model):   # Printing Model Size
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    
    
def load_config(filename):
    """
    Load the configuration from a YAML file and return a Config object.
    """
    class Config:
        def __init__(self, config_dict):
            self.__dict__.update(config_dict)

    # Load the YAML file and convert to Python object
    with open(filename, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = Config(config_dict)

    return(config)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
    torch.backends.cudnn.deterministic = True

    #this flag enables cudnn for some operations such as conv layers and RNNs, 
    # which can yield a significant speedup.
    torch.backends.cudnn.enabled = False

    # This flag enables the cudnn auto-tuner that finds the best algorithm to use
    # for a particular configuration. (this mode is good whenever input sizes do not vary)
    torch.backends.cudnn.benchmark = False


def main():
    # python3 main.py -no-delays 
    # python3 main.py -T 16 

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-no-delays', action='store_true')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-epochs', default=50, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('-data-dir', type=str, default = "DVS128Gesture", help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')

    args = parser.parse_args()


    # Model initialization 
    if args.no_delays:
        # No delays
        config = load_config('config_snn_no_delays.yaml')
        set_seed(config.seed)
        model = models.Net_No_Delays(config = config)
        optimizers = [torch.optim.Adam(model.parameters(), lr=config.lr_w, weight_decay=config.weight_decay)]
        lr_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) for optimizer in optimizers]

    else:
        # With delays
        config = load_config('config_snn_with_delays.yaml')
        set_seed(config.seed)
        model = models.Net_With_Delays(config=config)
        optimizers = [optim.Adam([{'params':model.weights, 'lr':model.config.lr_w, 'weight_decay':model.config.weight_decay},
                                                     {'params':model.weights_bn, 'lr':model.config.lr_w, 'weight_decay':0}]),
                    optim.Adam(model.positions, lr = model.config.lr_pos, weight_decay=0)]


        lr_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) for optimizer in optimizers]
    
    
    functional.set_step_mode(model, 'm')

    if args.cupy:
        functional.set_backend(model, 'cupy', instance=neuron.LIFNode)

    print(model)
    print_model_size(model)
    model.to(args.device)
    print("model runs on device:", args.device)

    # Data Loading
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=config.time_step, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=config.time_step, split_by='number')


    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1


    out_dir = os.path.join(args.out_dir, f'no-delays' if args.no_delays else 'delays' + f'T{config.time_step}_b{config.time_step}_lr{config.lr_w}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))



    ######################## Training loop ##########################
    
    for epoch in range(start_epoch, args.epochs):
        if not args.no_delays:
            model.collect_delays()
            model.draw_delays_all_evolution()
            model.draw_delays_all_evolution()

        start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in tqdm(train_data_loader):
            for optimizer in optimizers:
                optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = model(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = model(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            model.reset_model()

            if not args.no_delays:
                model.decrease_sig(epoch, args.epochs)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():

            if not args.no_delays:
                model.round_pos()

            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr = model(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                
                model.reset_model()

        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()