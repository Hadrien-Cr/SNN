from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from config import *

def DVS_dataloaders(config):
    train_dataset = DVS128Gesture(root="DVS128Gesture", train=True, data_type='frame', frames_number=config.time_step, split_by='number')
    test_dataset = DVS128Gesture(root="DVS128Gesture", train=False, data_type='frame', frames_number=config.time_step, split_by='number')

    train_loader = DataLoader(train_dataset,  batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=config.batch_size, num_workers=4)

    return train_loader, test_loader
if __name__ == '__main__':
    train_loader, test_loader = DVS_dataloaders(config)
    for i, (x, y) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        break