import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def add_noise(x):
    # print(x.shape)
    number_of_pixels = 100
    for i,j in zip(np.random.randint(0, 32, number_of_pixels),np.random.randint(0, 32, number_of_pixels)):
        x[:,:,i,j] = 0
    return x

def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "optimiser": opt.state_dict(),
        },
        os.path.join(save_dir, "models_%d.pth" % epoch),
    )

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_pretrain(models_path, net, device):
    model_dicts = torch.load(models_path, map_location=device)
    net.load_state_dict(model_dicts["state_dict"], strict=False)

def get_fraction_sampler(dataset, fraction):
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    split_index = int(np.floor(fraction / 100. * dataset_size))
    idx = dataset_indices[:split_index]
    return torch.utils.data.SubsetRandomSampler(idx)