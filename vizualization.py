import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE 
import seaborn as sns 

from pointSSL import POINT_SSL
from data import load_data, ModelNet


def __parse_args__():
    parser = argparse.ArgumentParser(description='Vizualization of embeddings')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of test batch, default 256)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Num of classes to vizualize')
    return parser.parse_args()

def main():

    args = __parse_args__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    point_ssl = POINT_SSL()
    point_ssl.load_state_dict(torch.load('checkpoints/models/point_ssl_1000.t7'), strict=False)

    train_points, train_labels = load_data("test")

    # Get indices of samples with labels in the first 10 classes
    indices_first_n_classes = np.where(train_labels < args.num_classes)[0]

    # Filter train_points and train_labels to only keep 5 first classes
    train_points = np.array([train_points[i] for i in indices_first_n_classes])
    train_labels = np.array([train_labels[i] for i in indices_first_n_classes])

    # get sample batch
    train_set = ModelNet(train_points, train_labels, set_type="test", num_points=2048) #TODO: replace with test
    loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size, shuffle=True)   
    sample = next(iter(loader))
    data, labels = sample

    labels = labels.cpu().detach().numpy()
    labels = labels.reshape(-1)


    # get representations
    point_ssl = point_ssl.half().to(device)
    point_ssl.eval()
    data = data.half().to(device)
    data = data.permute(0, 2, 1)
    embeddings, _ = point_ssl(data)
    embeddings = embeddings.cpu().detach().numpy()

    # get low dims tsne embeddings
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    # Plot
    ax =sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], alpha=0.5, hue=labels, palette="tab10")

    plt.savefig('plots/scatter_plot.png')
 
    

if __name__ == '__main__':
    main()
