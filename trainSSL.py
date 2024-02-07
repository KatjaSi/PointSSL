import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from pytorch_metric_learning.losses import NTXentLoss
from pointSSL import POINT_SSL
from pointSSL2 import POINT_SSL2
from data import  ModelNet, random_point_dropout, translate_pointcloud, load_data, load_data2

from agumentation import random_volume_crop_pc, rotate_pc


def train(model, projector, train_loader, criterion,  optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.half().to(device)
    model = nn.DataParallel(model)
    projector = projector.half().to(device)
    projector = nn.DataParallel(projector)


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
            x_prime_rep = model(x_prime)
            x_rep = model(x)
            x_prime_projection = projector(x_prime_rep)
            x_projection = projector(x_rep)
            projections = torch.cat((x_prime_projection, x_projection))
            indices = torch.arange(0, x_prime_projection.shape[0])
            labels = torch.cat((indices, indices))
            loss = criterion(projections, labels)
            loss.backward()
            optimizer.step()  
           # scheduler.step() TODO: where should it be?

            count += batch_size
            running_loss += loss.item() * batch_size
        scheduler.step()
        if (epoch % 100 == 0):
            torch.save(model.state_dict(), f'checkpoints/models/pct_simclr/pct_leiset_02{epoch+1}.t7')

        outstr = 'Epoch: %d, loss: %.6f' % (epoch, running_loss*1.0/count)
        print(outstr)



            
            

class ModelNetForSSL(Dataset):
    def __init__(self, data, num_points, crop_percentage=None, augmentation_strategy="strategy_2"):
        self.data = data
        self.num_points = num_points
        self.crop_percentage = crop_percentage
        self.augmentation_strategy = augmentation_strategy

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        x = self.data[idx][:self.num_points]
        x_prime = copy.deepcopy(x)

        if self.augmentation_strategy == "strategy_2":
            x = jitter_pointcloud(x)
            x = random_point_dropout(x) 
            x = translate_pointcloud(x)
            x = add_gaussian_noise(x)
            np.random.shuffle(x)

            x_prime = random_volume_crop_pc(x_prime, crop_percentage=random.uniform(0.1, 0.5)) #0.2,0.8
            x_prime = jitter_pointcloud(x_prime)
            x_prime = random_point_dropout(x_prime) 
            x_prime = translate_pointcloud(x_prime)
            x_prime = rotate_pointcloud(x_prime)
            x_prime = add_gaussian_noise(x_prime)
            np.random.shuffle(x_prime)

            return x_prime, x
        
        if self.augmentation_strategy == "strategy_1":
            x = random_point_dropout(x) 
            x = translate_pointcloud(x)
            np.random.shuffle(x)

            x_prime = random_volume_crop_pc(x_prime, crop_percentage=self.crop_percentage)
            x_prime = random_point_dropout(x_prime) 
            x_prime = translate_pointcloud(x_prime)
            np.random.shuffle(x_prime)

            return x_prime, x
        
        return x_prime, x

def main():
    train_points =  load_point_cloud_data_from_npy(data_dir="data/lei_dataset", num_files=20, num_points=2048)
    train_set = ModelNetForSSL(train_points, num_points=2048, augmentation_strategy="strategy_2")
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=256, shuffle=True)

    model = PCT_BASE(out_channels=512)

    loss = NTXentLoss(temperature = 0.2) #0.1
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    projector = MLPProjectionHead2(512, 256, 128)
    train(model, projector, train_loader, loss, optimizer, 401)
    torch.save(model.state_dict(), 'checkpoints/models/pct_simclr/pct_leiset.t7') # point_ssl_1000_8.t7 = strategy 2


if __name__ == '__main__':
    main()