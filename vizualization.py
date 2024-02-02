import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE 
import pandas as pd
import seaborn as sns 

from pointSSL import POINT_SSL
from siam import EncoderSiam
from data import load_data, ModelNet

from demo import SimpleModel


def __parse_args__():
    parser = argparse.ArgumentParser(description='Vizualization of embeddings')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of test batch, default 256)')
    parser.add_argument('--num_classes', type=int, default=40,
                        help='Num of classes to vizualize')
    return parser.parse_args()

def main():

    args = __parse_args__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    #point_ssl = POINT_SSL()
    #model = EncoderSiam()
    model = SimpleModel()
    #model.load_state_dict(torch.load('checkpoints/models/simsiam/simsiam_200.t7'), strict=False)


    train_points, train_labels = load_data("train")

    # Get indices of samples with labels in the first 10 classes
    indices_first_n_classes = np.where(train_labels < args.num_classes)[0]

    train_points = np.array([train_points[i] for i in indices_first_n_classes])
    train_labels = np.array([train_labels[i] for i in indices_first_n_classes])

    train_set = ModelNet(train_points, train_labels, set_type="test", num_points=2048) #should be test
    loader = DataLoader(dataset=train_set, num_workers=1, batch_size=512, shuffle=True)   
   
    embeddings_list = []
    labels_list = []

    model = model.half().to(device)
    model.eval()

    with torch.no_grad():
        for sample in loader:
            data, labels = sample
            data = data.half().to(device)
            data = data.permute(0, 2, 1)
            embeddings = model(data) # embeddings, _

            embeddings_list.append(embeddings.cpu().detach().numpy())
            labels_list.append(labels.cpu().detach().numpy())

    # Concatenate embeddings and labels across all batches
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    labels = labels.reshape(-1)

    # Get low-dimensional t-SNE embeddings
    embeddings_2d = TSNE(n_components=2, perplexity=60).fit_transform(embeddings)

    embeddings_2d = (embeddings_2d - np.mean(embeddings_2d, axis=0)) / np.std(embeddings_2d, axis=0)

    # Plot
    custom_palette10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    custom_palette20 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]
    ax = sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], alpha=0.5, hue=labels, palette=custom_palette10)
    ax.set_xlabel("TSNE Dimension 1")
    ax.set_ylabel("TSNE Dimension 2")
    plt.savefig('plots/scatter_plot.png')
    

if __name__ == '__main__':
    main()
