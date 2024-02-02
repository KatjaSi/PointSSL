import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PCT import PCT
from pointSSL import ProjectionHead #TODO: should projection include BN?
from data import load_data
from trainSSL import ModelNetForSSL
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

class MLP(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)

class EncoderSiam(nn.Module):
    def __init__(self, backbone=PCT(), projection=MLP()):
        super(EncoderSiam, self).__init__()
        self.backbone = backbone
        self.projection = projection
        #self.predictor = predictor

    def forward(self,x):
        x_repr = self.backbone(x)
        x_proj = self.projection(x_repr)
        return x_repr, x_proj


def D(p, z):
    z = z.detach() # stop gradient
    #p = F.normalize(p, p=2, dim=1)  # L2-normalize
    p = nn.functional.normalize(p, dim=-1)
    #z = F.normalize(z, p=2, dim=1)  # L2-normalize
    z = nn.functional.normalize(z, dim=-1)
    return -(p * z).sum(dim=1).mean()


def loss(p1,p2,z1,z2):
    return D(p1,z2)/2 + D(p2,z1)/2



def train(model, train_loader:DataLoader, predictor, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model= model.half().to(device)
    model = nn.DataParallel(model)

    predictor = predictor.half().to(device)
    predictor = nn.DataParallel(predictor)

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

            _, z1 = model(x_prime)
            _, z2 = model(x)
            p1, p2 = predictor(z1), predictor(z2)
            loss = criterion(p1,p2,z1,z2)
            loss.backward()
            optimizer.step()

            count += batch_size
            running_loss += loss.item() * batch_size

        outstr = 'Epoch: %d, loss: %.6f' % (epoch+1, running_loss*1.0/count)
        print(outstr)
        scheduler.step()
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints/models/simsiam/simsiam_{epoch+1}.t7')   


           

if __name__ == '__main__':
    train_points, _ = load_data("train")
    test_points, _ = load_data("test")
    train_set = ModelNetForSSL(train_points, num_points=2048, augmentation_strategy="strategy_2")
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=64, shuffle=True)

    pct = PCT(out_channels=512)
    ph = MLP()

    predictor = MLP(in_size=256, out_size=256, hidden_size=512)
    f = EncoderSiam(backbone=pct, projection=ph)
    opt = optim.SGD(pct.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4)

    #opt = torch.optim.Adam(pct.parameters(), lr=1e-3)

    train(f, train_loader, predictor, loss, opt, 1001)
