import torch
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from data import  ModelNet, ModelNetForMaskedLearning
from models.transformers import PCT_Decoder, PCT_BASE2, PCT_BASE
from models.models import PointCloudAE, PointCloudAEEncoder, PointCloudAEDecoder
from models.linear_models import PointCloudDecoder
from data import load_data
#from mask_loss.extensions.chamfer_dist import ChamferDistanceL2 as chamfer_loss
from mask_loss.extensions.emd.emd import EarthMoverDistance as emd_loss
#from criterion import sinkhorn_loss

if __name__ == '__main__':

    train_points, train_labels = load_data("train", "data/modelnet40_ply_hdf5_2048")
    #train_set = ModelNet(train_points, train_labels, num_points=2048, set_type="train") # set type = train
    train_set = ModelNetForMaskedLearning(train_points, num_points=1024, rotate=True, num_key_points=1, num_neighbors=1)
    train_loader = DataLoader(
            dataset=train_set,
            num_workers=1,
            batch_size=16, #32
            shuffle=True)

    test_points, test_labels = load_data("test", "data/modelnet40_ply_hdf5_2048")
    #test_set = ModelNet(test_points, test_labels, num_points=2048, set_type="test")
    test_set = ModelNetForMaskedLearning(test_points, num_points=1024, rotate=True, mask=False, num_key_points=1, num_neighbors=10)
    test_loader = DataLoader(
            dataset=test_set,
            num_workers=1,
            batch_size=16,
            shuffle=False)


    encoder = PCT_BASE(out_dim=512)
    #encoder = PointCloudAEEncoder(1024, 128)
    device = torch.device("cuda")
    encoder = encoder.to(device)
    #decoder = PCT_Decoder(input_dim=128)
    decoder = PointCloudAEDecoder(1024, 512) 
    decoder = decoder.to(device)

    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
   # optimizer = SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01, momentum=0.9)
    
    #criterion = chamfer_loss()
    criterion = emd_loss()

    num_epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=num_epochs, 
                                                        eta_min=optimizer.param_groups[0]['lr']/100)
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        for data in train_loader:
            x_aug = data['augmented']
            x = data['original']
            mask = data['mask']
            x_aug = x_aug.to(device)
            x = x.to(device)
            mask = mask.to(device)  
            latent = encoder(x_aug.permute(0,2,1)) #, mask=mask)
            reconstructed = decoder(latent)
            loss = criterion(reconstructed, x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}]:\nTrain Loss: {epoch_loss:.8f}')
        scheduler.step()

        # test part
        encoder.eval()
        decoder.eval()

        test_loss = 0
        for data in test_loader:
            x_aug = data['augmented']
            x = data['original']
            mask = data['mask']
            #x, _ = data
            x = x.to(device)
            x_aug = x_aug.to(device)
            mask = mask.to(device)
            latent = encoder(x_aug.permute(0,2,1)) #, mask=mask)
            #_, lf = encoder(x_aug.permute(0,2,1)) #, mask=mask)
            reconstructed = decoder(latent)
            test_loss += criterion(reconstructed, x).mean().item()
        epoch_loss = test_loss / len(test_loader)
        print(f'Test Loss: {epoch_loss:.8f}')



    
    encoder.eval()
    decoder.eval()
    it = iter(test_loader)
    next(it)
    data = next(it)
    x_aug = data['augmented']
    x = data['original']
    mask = data['mask']
    x_aug = x_aug.to(device)
    mask = mask.to(device)
    #gf, lf = encoder(data.permute(0,2,1)) 
    #latent = encoder(x_aug.permute(0,2,1), mask=mask)
    latent = encoder(x.to(device).permute(0,2,1))
    reconstructed = decoder(latent).cpu().detach().numpy()
    r_pc = reconstructed[4]  # 20
    fig = go.Figure(data=[go.Scatter3d(x=r_pc[:, 0], 
                                    y=r_pc[:, 1], 
                                   z=r_pc[:, 2], 
                                    mode='markers',
                                   marker=dict(size=5))])
    fig.write_html('reconstructed.html')

    o_pc = x[4].cpu().detach().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=o_pc[:, 0], 
                                    y=o_pc[:, 1], 
                                   z=o_pc[:, 2], 
                                    mode='markers',
                                   marker=dict(size=5))])
    fig.write_html('original.html')

    a_pc = x_aug[4].cpu().detach().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=a_pc[:, 0], 
                                    y=a_pc[:, 1], 
                                   z=a_pc[:, 2], 
                                    mode='markers',
                                   marker=dict(size=5))])
    fig.write_html('augmented.html')



