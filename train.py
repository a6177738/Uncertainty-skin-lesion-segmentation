import torch
import time
from torch.autograd import Variable
from torchvision import transforms
import model
from demo2 import evaluate
from  test_isic17 import  evalua
from data import Data
from optimizer import PolyOptimizer
from loss import SimMaxLoss,SimMinLoss

batch_size = 64
num_worker = 8
momentum = 0.9
weight_decay = 0.0005
epochs  = 300
root= "data\\ISIC-2017\\"
ground_path = "ISIC-2017_Training_Part1_GroundTruth\\"
data_path = "ISIC-2017_Training_Data\\"
if __name__=='__main__':

    train_transforms = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    ])
    dataset = Data(root,data_path,ground_path,train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
    )
    Net = model.get_model(pretrained="mocov2").cuda()
    Net.train()
    path_model = "checkpoints\\WSSS\\moco-alpha-0.25-bs128.pth"
    Net.load_state_dict(torch.load(path_model))

    param_groups = Net.get_parameter_groups()

    criterion = [SimMaxLoss(metric='cos', alpha=0.25), SimMinLoss(metric='cos'),
                 SimMaxLoss(metric='cos', alpha=0.25)]

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': 0.001, 'weight_decay': 1e-4},
        {'params': param_groups[1], 'lr': 2 * 0.001, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * 0.001, 'weight_decay': 1e-4},
        {'params': param_groups[3], 'lr': 20 * 0.001, 'weight_decay': 0},
    ], lr=0.001, momentum=0.9, weight_decay=1e-4, max_step=60000)
    for epoch in range(epochs):
        for batch_i, (img, img_name) in enumerate(dataloader):
            start_time = time.time()
            batches_done = batch_i + 1
            img = Variable(img.float()).cuda()

            optimizer.zero_grad()
            fg_feats, bg_feats, ccam = Net(img)
            end_time = time.time()
            t = end_time - start_time

            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)

            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            print("epoch:", epoch, " batch_i:", batch_i, " time:", t, " loss:", loss)
        parapth = "check\\" + str(epoch) + ".pth"
        torch.save(Net.state_dict(), parapth)


