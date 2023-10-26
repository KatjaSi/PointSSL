import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from pytorch_metric_learning.losses import NTXentLoss
from pointSSL import POINT_SSL
from data import  ModelNet, random_point_dropout, translate_pointcloud, load_data

from agumentation import random_volume_crop_pc


def train(model:POINT_SSL, train_loader:DataLoader, criterion,  optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.half().to(device)
    model = nn.DataParallel(model)

    learning_rate = optimizer.param_groups[0]['lr']
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=learning_rate/100)

    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0.0
        model.train()
        for x_prime, x in (train_loader):
            batch_size = len(x_prime)
            x_prime = x_prime.half().to(device)
            x = x.half().to(device)
            x = x.permute(0, 2, 1)
            x_prime = x_prime.permute(0,2,1)
            optimizer.zero_grad()
            x_prime_rep, x_rep, x_prime_projection, x_projection = model(x_prime, x)
            projections = torch.cat((x_prime_projection, x_projection))
            indices = torch.arange(0, x_prime_projection.shape[0])
            labels = torch.cat((indices, indices))
            loss = criterion(projections, labels)
            loss.backward()
            optimizer.step()  
            scheduler.step()

            count += batch_size
            running_loss += loss.item() * batch_size

        outstr = 'Epoch: %d, loss: %.6f' % (epoch, running_loss*1.0/count)
        print(outstr)


            
            

class ModelNetForSSL(Dataset):
    def __init__(self, data, num_points=1024, crop_percentage=0.4):
        self.data = data
        self.num_points = num_points
        self.crop_percentage = crop_percentage

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        x = self.data[idx][:self.num_points]
        x_prime = copy.deepcopy(x)

        x = random_point_dropout(x) 
        x = translate_pointcloud(x)
        np.random.shuffle(x)

        x_prime = random_volume_crop_pc(x_prime, crop_percentage=self.crop_percentage)
        x_prime = random_point_dropout(x_prime) 
        x_prime = translate_pointcloud(x_prime)
        np.random.shuffle(x_prime)

        return x_prime, x

def main():
    print(torch.cuda.is_available())
    return
    train_points, _ = load_data("train")
    test_points, _ = load_data("test")
    train_set = ModelNetForSSL(combined_train_points, num_points=2048, crop_percentage=0.3)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=512, shuffle=True)

    model = POINT_SSL()

    loss = NTXentLoss(temperature = 0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    train(model, train_loader, loss, optimizer,1000)

    torch.save(model.state_dict(), 'checkpoints/models/point_ssl.t7')


if __name__ == '__main__':
    main()