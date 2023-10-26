import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import sklearn.metrics as metrics
import numpy as np
from torch.utils.data import DataLoader
from criterion import cross_entropy_loss_with_label_smoothing
from pointSSL import POINT_SSL
from data import  ModelNet

from data import load_data
seed = 42
random.seed(seed)

def __parse_args__():
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--saved_model', type=str, default=256,
                        help='Name of saved model')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of test batch, default 256)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to sample from each point cloud, should be the same as for training')
    return parser.parse_args()

def test(model, test_loader:DataLoader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.half().to(device)
   

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


    outstr = 'Test acc: %.6f, test avg acc: %.6f' % (test_acc,avg_per_class_acc)
    print(outstr)


def main():
    
    args = __parse_args__()
    test_points, test_labels = load_data("test")
    test_set = ModelNet(test_points, test_labels, set_type="test", num_points=args.num_points)
 
    pct = POINT_SSL(output_channels=40)

    pct = nn.DataParallel(pct)

    pct.load_state_dict(torch.load(f'checkpoints/models/{args.saved_model}'), strict=False)

    test_loader = DataLoader(
                    dataset=test_set, 
                    num_workers=4,
                    batch_size=args.batch_size, 
                    shuffle=False,  
                    worker_init_fn=lambda x: torch.manual_seed(seed))

    test(  model=pct,
            test_loader=test_loader,
            criterion=cross_entropy_loss_with_label_smoothing)


if __name__ == '__main__':
    main()