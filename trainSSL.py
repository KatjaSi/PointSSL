import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from pytorch_metric_learning.losses import NTXentLoss
from pointSSL import POINT_SSL
from pointSSL2 import POINT_SSL2
from data import  ModelNet, random_point_dropout, translate_pointcloud, load_data, ModelNetAugmented, load_point_cloud_data_from_npy
from data import ModelNetForSSL
from models.transformers import PCT_BASE2
from models.linear_models import MLPProjectionHead2



def train(model, projector, train_loader, criterion,  optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.half().to(device)
    #model = nn.DataParallel(model)
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
        if (epoch % 50 == 0):
            torch.save(model.state_dict(), f'checkpoints/models/pct_simclr/pct_base2_leiset{epoch+1}.t7')

        outstr = 'Epoch: %d, loss: %.6f' % (epoch, running_loss*1.0/count)
        print(outstr)



def main():
    train_points =  load_point_cloud_data_from_npy(data_dir="data/lei_dataset", num_files=20, num_points=1024)
    train_set = ModelNetForSSL(train_points, num_points=1024)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=512, shuffle=True)

    model = PCT_BASE2(out_channels=128)
    model = nn.DataParallel(model)
    sd = torch.load("checkpoints/models/pct_simclr/pct_base2_leiset51.t7")
    model.load_state_dict(sd)

    loss = NTXentLoss(temperature = 0.1) #0.1
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    projector = MLPProjectionHead2(128, 64, 32)
    train(model, projector, train_loader, loss, optimizer, 401)
    torch.save(model.state_dict(), 'checkpoints/models/pct_simclr/pct_leiset.t7') # point_ssl_1000_8.t7 = strategy 2


if __name__ == '__main__':
    main()