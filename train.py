import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import sklearn.metrics as metrics
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from criterion import cross_entropy_loss_with_label_smoothing
from pointSSL import POINT_SSL
from data import  ModelNet

from data import load_data, load_data2
seed = 42
random.seed(seed)

def train(model:POINT_SSL, train_loader:DataLoader, test_loader:DataLoader, criterion, optimizer, num_epochs, pretrained):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.half().to(device)
    model = nn.DataParallel(model)

    learning_rate = optimizer.param_groups[0]['lr']
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=learning_rate/100)

    best_test_acc = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0.0
        model.train()

        train_pred = []
        train_true = []
        idx = 0
        for data, labels in (train_loader):
            batch_size = len(labels)
            data = data.half().to(device) 
            labels = labels.to(device)
            data = data.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(data, downstream=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = outputs.max(dim=1)[1]
            count += batch_size
            running_loss += loss.item() * batch_size
            train_true.append(labels.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Epoch: %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                    running_loss*1.0/count,
                                                                                    metrics.accuracy_score(
                                                                                    train_true, train_pred),
                                                                                    metrics.balanced_accuracy_score(
                                                                                    train_true, train_pred))
        print(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, labels in (test_loader):
            data, labels = data.half().to(device), labels.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            outputs = model(data, downstream=True)
            loss = criterion(outputs, labels)
            preds = outputs.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        outstr = 'Epoch: %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        print(outstr)
        print(f"best test accuracy is {best_test_acc}")
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            if not pretrained:
                #torch.save(model.state_dict(), 'checkpoints/models/pointSSL_without_pretraining_250_3.t7') # 91.57 % now
                pass
            else:
               # torch.save(model.state_dict(), 'checkpoints/models/pointSSL_with_pretraining_250_3.t7')
               pass
    
    print(f"Finished Training, best test accuracy is {best_test_acc}")

def __parse_args__():
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of batch, default 256)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='Number of training epochs, default 250')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='Learning rate (0.01 by defaukt)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to sample from each point cloud')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--pretrained', action='store_true')
    return parser.parse_args()


def reduce_data(data, labels, percentage=10):
    # Calculate the number of samples to keep
    num_samples_to_keep = int(len(data) * (percentage / 100))

    # Randomly sample data and labels
    sampled_indices = random.sample(range(len(data)), num_samples_to_keep)
    reduced_data = [data[i] for i in sampled_indices]
    reduced_labels = [labels[i] for i in sampled_indices]
    return reduced_data, reduced_labels

def main():
    
    args = __parse_args__()
    #train_points, train_labels = load_data("train")
    #test_points, test_labels = load_data("test")
    train_points, train_labels = load_data2("train", folder="shapenetcorev2_hdf5_2048")
    test_points, test_labels = load_data2("test",  folder="shapenetcorev2_hdf5_2048")

  #  reduced_train_points, reduced_train_labels = reduce_data(train_points, train_labels, percentage=10)


    train_set = ModelNet(train_points, train_labels, set_type="train", num_points=args.num_points) #reduced_train_points
    test_set = ModelNet(test_points, test_labels, set_type="test", num_points=args.num_points)
 
    pct = POINT_SSL(output_channels=55) #40
    # pretrained or not?
    if args.pretrained:
       # pct.load_state_dict(torch.load('checkpoints/models/point_ssl_1000_8.t7'), strict=False) # point_ssl_1000_8 = strategy2
       state_dict = torch.load('checkpoints/models/point_ssl_1000_8.t7')
       #print(state_dict.keys())
       del state_dict['projection.linear1.weight']
       del state_dict['projection.linear1.bias']
       del state_dict['projection.linear2.weight']
       del state_dict['projection.linear2.bias']
       del state_dict['linear4.weight']
       del state_dict['linear4.bias']

        # Load the modified state dictionary into the model, setting strict to False
       pct.load_state_dict(state_dict, strict=False)
    

    # Set batch size
    batch_size = args.batch_size

    # Create DataLoader instances
    train_loader = DataLoader(
                    dataset=train_set,
                    num_workers=4,
                    batch_size=batch_size,
                    shuffle=True,
                    worker_init_fn=lambda x: torch.manual_seed(seed))
    test_loader = DataLoader(
                    dataset=test_set, 
                    num_workers=4,
                    batch_size=batch_size, 
                    shuffle=False,  
                    worker_init_fn=lambda x: torch.manual_seed(seed))

    opt = optim.SGD(pct.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4) 
    train(  model=pct,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=cross_entropy_loss_with_label_smoothing,
            optimizer=opt,
            num_epochs=args.epochs,
            pretrained=args.pretrained
            )

if __name__ == '__main__':
    main()