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
import argparse
from model import NetClassification
from helpers import imshow, save_ckpt, load_pretrain, get_fraction_sampler

parser = argparse.ArgumentParser(description="AutoBots")
parser.add_argument("--slurm", type=str, default="", help="Experiment identifier")
parser.add_argument("--models_path", type=str, default="", help="Load model checkpoint")
parser.add_argument("--freeze_backbone", type=bool, default=False, help="f")
parser.add_argument("--fraction", type=int, default=100, help="%, Percentage of how much data is used")
args = parser.parse_args()


# code refer to the pytorch tutorial
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
if args.fraction != 100:
    train_sampler = get_fraction_sampler(trainset, args.fraction)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2, sampler=train_sampler)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
if args.fraction != 100:
    test_sampler = get_fraction_sampler(testset, args.fraction)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, sampler=test_sampler)
else:
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train_cls_epoch(net, device, dataloader, loss_fn, optimizer, num_sample):
    # Set train mode
    net.train()
    running_corrects = 0
    running_loss = 0.0
    
    for data in dataloader:
        # Move tensor to the proper device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.set_grad_enabled(True):
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Evaluate loss
            loss = loss_fn(outputs, labels)
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / num_sample
    epoch_acc = running_corrects.double() / num_sample
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', flush=True)
    return epoch_acc

def test_cls_epoch(net, device, dataloader, loss_fn, optimizer, num_sample):
    # Set train mode
    net.eval()
    running_corrects = 0
    running_loss = 0.0
    
    for data in dataloader:
        # Move tensor to the proper device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            # Evaluate loss
            loss = loss_fn(outputs, labels)

        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * inputs.size(0)
    
        # Backward pass
        optimizer.zero_grad()
        
    epoch_loss = running_loss / num_sample
    epoch_acc = running_corrects.double() / num_sample
    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', flush=True)
    return epoch_acc


# refer the code from pytorch tutorial
def visualize_model(model, testloader, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(1, num_images, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {classes[preds[j]]}')
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return
        model.train(mode=was_training)
    
    
### Define an optimizer (both for the encoder and the decoder!)
lr= 0.01
d = 8


net_cls = NetClassification(encoded_space_dim=d,fc2_input_dim=128, num_class=10)
net_cls.to(device)
optim_cls = torch.optim.Adam(net_cls.parameters(), lr=lr, weight_decay=1e-05)
loss_fn_cls = torch.nn.CrossEntropyLoss()

if args.models_path:
    load_pretrain(args.models_path, net_cls, device)

if args.freeze_backbone:
    for param in net_cls.encoder.parameters():
        param.requires_grad = False

    for param in net_cls.decoder.parameters():
        param.requires_grad = False

num_epochs = 40
save_dir = os.path.join('./cifar_pth', args.slurm)

best_val_acc = 0

val_acc = test_cls_epoch(net_cls, device,testloader,loss_fn_cls, optim_cls, len(testset))
print('Before training: val_acc ', val_acc, flush=True)

for epoch in range(num_epochs):
    print('\n EPOCH {}/{} '.format(epoch + 1, num_epochs), flush=True)
    
    train_cls_epoch(net_cls, device, trainloader, loss_fn_cls, optim_cls, len(trainset))
    val_acc = test_cls_epoch(net_cls, device,testloader,loss_fn_cls, optim_cls, len(testset))
    
    if val_acc > best_val_acc:
        save_ckpt(net_cls, optim_cls, save_dir, epoch)
        best_val_acc = val_acc
    # visualize_model(net_cls, testloader, device)
    if epoch == 20:
        print("best_val_acc 20", best_val_acc, flush=True)

print("best_val_acc", best_val_acc, flush=True)