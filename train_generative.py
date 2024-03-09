import torch
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from data import  ModelNet, ModelNetForMaskedLearning
from models.transformers import PCT_ml, PCT_Decoder, PCT_BASE2
from models.models import PointCloudAE, PointCloudAEEncoder, PointCloudAEDecoder
from models.linear_models import PointCloudDecoder
from data import load_data
from mask_loss.extensions.chamfer_dist import ChamferDistanceL2 as chamfer_loss

if __name__ == '__main__':

    train_points, train_labels = load_data("train", "data/modelnet40_ply_hdf5_2048")
    #train_set = ModelNet(train_points, train_labels, num_points=2048, set_type="train") # set type = train
    train_set = ModelNetForMaskedLearning(train_points, num_points=1024, rotate=False, mask=True, num_key_points=5, num_neighbors=100)
    train_loader = DataLoader(
            dataset=train_set,
            num_workers=2,
            batch_size=32,
            shuffle=True)

    test_points, test_labels = load_data("test", "data/modelnet40_ply_hdf5_2048")
    #test_set = ModelNet(test_points, test_labels, num_points=2048, set_type="test")
    test_set = ModelNetForMaskedLearning(test_points, num_points=1024, rotate=False, mask=True, num_key_points=5, num_neighbors=100)
    test_loader = DataLoader(
            dataset=test_set,
            num_workers=2,
            batch_size=32,
            shuffle=False)


    encoder = PCT_BASE2(out_channels=128)
    #encoder = PCT_ml(out_dim=128, mask=True)
    #encoder = PointCloudAEEncoder(1024, 128)
    device = torch.device("cuda")
    encoder = encoder.to(device)
    #decoder = PCT_Decoder(input_dim=128)
    decoder = PointCloudAEDecoder(1024, 128*2)
    #decoder = PointCloudDecoder()
    decoder = decoder.to(device)

    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    
    criterion = chamfer_loss()

    #model = PointCloudAE(1024, 128)
    #model = model.to(device)
    #optimizer = Adam(model.parameters(), lr=0.0005)

    num_epochs = 100
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
            #print((x_aug[0].cpu() == torch.tensor([0.0, 0.0, 0.0])).all(dim=1).sum().item())
            latent = encoder(x_aug.permute(0,2,1)) #, mask=mask)
            #_, lf = encoder(x_aug.permute(0,2,1))#, mask=mask)
            reconstructed = decoder(latent)
            loss = criterion(reconstructed, x)
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
            test_loss += criterion(reconstructed, x).item()
        epoch_loss = test_loss / len(test_loader)
        print(f'Test Loss: {epoch_loss:.8f}')



    
    encoder.eval()
    decoder.eval()
    data = next(iter(test_loader))
    x_aug = data['augmented']
    x = data['original']
    mask = data['mask']
    x_aug = x_aug.to(device)
    mask = mask.to(device)
    #gf, lf = encoder(data.permute(0,2,1)) 
    latent, _= encoder(x_aug.permute(0,2,1), mask=mask)
    reconstructed = decoder(latent).cpu().detach().numpy()
    r_pc = reconstructed[20]
    fig = go.Figure(data=[go.Scatter3d(x=r_pc[:, 0], 
                                    y=r_pc[:, 1], 
                                   z=r_pc[:, 2], 
                                    mode='markers',
                                   marker=dict(size=5))])
    fig.write_html('reconstructed.html')

    o_pc = x[20].cpu().detach().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=o_pc[:, 0], 
                                    y=o_pc[:, 1], 
                                   z=o_pc[:, 2], 
                                    mode='markers',
                                   marker=dict(size=5))])
    fig.write_html('original.html')

    a_pc = x_aug[20].cpu().detach().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=a_pc[:, 0], 
                                    y=a_pc[:, 1], 
                                   z=a_pc[:, 2], 
                                    mode='markers',
                                   marker=dict(size=5))])
    fig.write_html('augmented.html')



