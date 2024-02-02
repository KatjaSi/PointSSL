import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from data import load_data, ModelNet
from PCT import PCT
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from pointSSL import POINT_SSL
from pointSSL2 import POINT_SSL2

if __name__ == '__main__':
    train_points, train_labels = load_data("train")
    test_points, test_labels = load_data("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    #pct = POINT_SSL2()
    pct = PCT() 
    pct.load_state_dict(torch.load('checkpoints/models/point_ssl_1000_8.t7'), strict=False)
    #pct.load_state_dict(torch.load('checkpoints/models/simsiam/simsiam_200.t7'), strict=False)
    pct.eval()

    train_set = ModelNet(train_points, train_labels, set_type="test", num_points=2048)

    loader = DataLoader(
                    dataset=train_set, 
                    num_workers=1,
                    batch_size=256, 
                    shuffle=False)   

    representations = []
    labels = []
    for data, label in (loader):
        data = data.half().to(device) 
        label = label.to(device)
        data = data.permute(0, 2, 1)
        pct = pct.half().to(device)
        pct = nn.DataParallel(pct)
        representations.append(pct(data).detach().cpu().numpy())
        labels.extend(label.cpu().numpy())
        #break # trying on one batch now

    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(representations)
    

    svm_classifier = SVC(kernel="linear", random_state=0)
    svm_classifier.fit(X_train, labels.ravel())

    # prediction of test representations
    test_set = ModelNet(test_points, test_labels, set_type="test", num_points=2048)

    test_loader = DataLoader(
                    dataset=test_set, 
                    num_workers=1,
                    batch_size=128, 
                    shuffle=False)   

    representations = []
    labels = []
    for data, label in (test_loader):
        data = data.half().to(device) 
        label = label.to(device)
        data = data.permute(0, 2, 1)
        pct = pct.half().to(device)
        pct = nn.DataParallel(pct)
        representations.append(pct(data).detach().cpu().numpy())
        labels.extend(label.cpu().numpy())


    representations = np.concatenate(representations, axis=0)

    labels = np.concatenate(labels, axis=0)

    X_test = sc.transform(representations)

    y_pred = svm_classifier.predict(X_test)
    #print(y_pred[:10])
    #print(labels.ravel()[:10])
    print(accuracy_score(labels.ravel(), y_pred))

