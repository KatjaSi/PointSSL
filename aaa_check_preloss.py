import torch
#from mask_loss.extensions.emd.emd import earth_mover_distance as emd_loss

def pc_loss(loss_name):     # batch, length, dim
    if loss_name == 'EMD':
        from mask_loss.extensions.emd.emd import earth_mover_distance as emd_loss
        point_loss = emd_loss()
    elif loss_name == 'CD':
        from mask_loss.extensions.chamfer_dist import ChamferDistanceL2 as chamfer_loss
        point_loss = chamfer_loss()
    else:
        raise Exception(f"No such loss: {loss_name}")
    return point_loss

batch = 64
dim = 3

length1 = 128
a1 = torch.ones(batch, length1, dim)
b1 = torch.ones(batch, length1, dim) * 2

length2 = 1024
a2 = torch.ones(batch, length2, dim) * 4
b2 = torch.ones(batch, length2, dim) * 2

#point_loss_emd = pc_loss('EMD')
point_loss_cd = pc_loss('CD')

a1, b1, a2, b2 = a1.cuda(), b1.cuda(), a2.cuda(), b2.cuda()
a1.requires_grad = True
b1.requires_grad = True
a2.requires_grad = True
b2.requires_grad = True
#loss_emd1 = point_loss_emd(a1, b1)
#loss_emd2 = point_loss_emd(a2, b2)
loss_cd1 = point_loss_cd(a1, b1)
loss_cd2 = point_loss_cd(a2, b2)

#print(loss_emd1, loss_emd2)
print(loss_cd1, loss_cd2)

#loss_emd1.backward()
#loss_emd2.backward()
loss_cd1.backward()
loss_cd2.backward()

#loss_emd3 = point_loss_emd(a1, a2)
#print(loss_emd3)
loss_cd3 = point_loss_cd(a1, a2)
print(loss_cd3)

# print(a1.dtype, a1.device)
# af = a1.float()
# print(af.dtype, af.device)
