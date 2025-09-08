import os
from PIL import Image
import io

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset


class Wind(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train

        self.speed_mean = 7.552827
        self.speed_std = 3.933559
        self.direction_mean = 203.931985
        self.direction_std = 81.718628

        this_dir = os.path.dirname(os.path.realpath(__file__))

        self.data_root = os.path.join(this_dir, '../../tools/data/wind_15_19-22')

        if is_train:
            with open(os.path.join(self.data_root, 'train_set.txt')) as f:
                self.dataset = eval(f.read())
        else:
            with open(os.path.join(self.data_root, 'val_set.txt')) as f:
                self.dataset = eval(f.read())

        self.mean = 0
        self.std = 1
        self.data_name = 'wind'

    def __getitem__(self, idx):
        x = torch.tensor([  # (T, 2)
            [float(elem[1]), float(elem[2])] for elem in self.dataset[idx][:144]
        ])
        x[:, 0] = (x[:, 0] - self.speed_mean) / self.speed_std
        x[:, 1] = (x[:, 1] - self.direction_mean) / self.direction_std

        y = torch.tensor([  # (T, 2)
            [float(elem[1]), float(elem[2])] for elem in self.dataset[idx][144:]
        ])
        y[:, 0] = (y[:, 0] - self.speed_mean) / self.speed_std
        y[:, 1] = (y[:, 1] - self.direction_mean) / self.direction_std

        year, month, day = self.dataset[idx][0][0][:10].split('-')

        return x, y, int(year), int(month), int(day)

    def __len__(self):
        return len(self.dataset)


def load_data(batch_size, val_batch_size, num_workers=4):

    train_set = Wind(is_train=True)
    valid_set = Wind(is_train=False)

    dataloader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    dataloader_vali = torch.utils.data.DataLoader(
        valid_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False
    )
    dataloader_test = torch.utils.data.DataLoader(
        valid_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False
    )

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataloader_train, _, dataloader_test = load_data(batch_size=4, val_batch_size=4, num_workers=4)

    print(len(dataloader_train), len(dataloader_test))
    print(len(dataloader_train.dataset), len(dataloader_test.dataset))

    for i, item in enumerate(dataloader_train):
        print(item[0].shape, item[1].shape)
        # print(item[2], item[3], item[4])
        # print(item[0][0, :, 0], item[1][0, :, 0])
        # print(item[0][0, :, 1], item[1][0, :, 1])
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(item[0][0, :, 0])
        axs[0, 0].set_title('x wind_speed')

        axs[0, 1].plot(item[0][0, :, 1])
        axs[0, 1].set_title('x wind_direction')

        axs[1, 0].plot(item[1][0, :, 0])
        axs[1, 0].set_title('y wind_speed')

        axs[1, 1].plot(item[1][0, :, 1])
        axs[1, 1].set_title('y wind_direction')
        plt.show()

        # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        # axs[0].plot(item[0][0, :, 0] * 3.933559 + 7.552827)
        # axs[0].set_title('wind_speed(m/s)')
        #
        # axs[1].plot(item[0][0, :, 1] * 81.718628 + 203.931985)
        # axs[1].set_title('wind_direction(°)')
        # plt.show()

        if i >= 0:
            break

    for i, item in enumerate(dataloader_test):
        print(item[0].shape, item[1].shape)
        # print(item[2], item[3], item[4])
        # print(item[0][0, :, 0], item[1][0, :, 0])
        # print(item[0][0, :, 1], item[1][0, :, 1])
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(item[0][0, :, 0])
        axs[0, 0].set_title('x wind_speed')

        axs[0, 1].plot(item[0][0, :, 1])
        axs[0, 1].set_title('x wind_direction')

        axs[1, 0].plot(item[1][0, :, 0])
        axs[1, 0].set_title('y wind_speed')

        axs[1, 1].plot(item[1][0, :, 1])
        axs[1, 1].set_title('y wind_direction')
        plt.show()

        if i >= 0:
            break