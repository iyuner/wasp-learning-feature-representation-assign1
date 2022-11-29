import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model import Net
from helpers import add_noise, save_ckpt, load_pretrain
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser(description="AutoBots")
parser.add_argument("--slurm", type=str, default="", help="Experiment identifier")
args = parser.parse_args()

filename = "image_tensor.bin"
img = np.memmap(filename, dtype='uint8',shape=(100000,3,32,32))
img = img/256 # normalize between 0 -1 

data = torch.Tensor(img)
dataset = TensorDataset(data) # create your datset

train_transform = transforms.Compose([
    transforms.ToTensor()
])
dataset.transform = train_transform
batch_size = 256
train_set, val_set = torch.utils.data.random_split(dataset, [90000, 10000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.01

### Set the random seed for reproducible results
# torch.manual_seed(0)

### Initialize the two networks
d = 8

net = Net(encoded_space_dim=d,fc2_input_dim=128)

optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-05)

# Move both the encoder and the decoder to the selected device
net.to(device)


### Training function
def train_epoch(net, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    net.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        # print(image_batch)
        image_batch = image_batch[0].to(device)
        image_batch = add_noise(image_batch)
        data = net(image_batch)
        # Evaluate loss
        loss = loss_fn(data, image_batch) *256
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def plot_ae_outputs(net,n=10):
    figure = plt.figure(figsize=(16,4.5))
    # targets = val_set.numpy()
    t_idx = np.random.randint(0,9000,size=n)
    idx = 0 
    for i in t_idx:
        ax = plt.subplot(2,n,idx+1)
        img = val_set[i][0].unsqueeze(0).to(device)
        img = add_noise(img)
        net.eval()
        with torch.no_grad():
            rec_img  = net(img)
        plt.imshow(img.cpu().squeeze().numpy().T) # .astype('uint8')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if idx == n//2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, idx + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy().T) # .astype('uint8')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if idx == n//2:
            ax.set_title('Reconstructed images')
        idx += 1
    plt.show() 
    # return figure

writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), args.slurm, "tb_files"))

"""visualize the reconstruction figure
models_path = 'pth/models_570.pth'
load_pretrain(models_path, net, device)
plot_ae_outputs(net,n=10)
"""


num_epochs = 10000
diz_loss = {'train_loss':[],'val_loss':[]}
save_dir = './pth'
best_loss = 20
for epoch in range(num_epochs):
    train_loss = train_epoch(net, device, train_loader, loss_fn, optim)
    # val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
    print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss), flush=True)
    diz_loss['train_loss'].append(train_loss)
    # diz_loss['val_loss'].append(val_loss)
    if epoch % 10 == 0 and epoch > 0:
        figure = plot_ae_outputs(net,n=10)
        writer.add_figure('figure', figure, epoch) 
        if train_loss < best_loss:
            save_ckpt(net, optim, save_dir, epoch)
            best_loss = train_loss
    
    writer.add_scalar("train_loss", train_loss.item(), epoch)

    

