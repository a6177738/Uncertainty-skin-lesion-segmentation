import torch
from model import ResFCN
import time
from data_nousl17 import Data
from torch.autograd import Variable
from transform import *

batch_size = 16
num_worker = 8
lr = 4e-3
momentum = 0.9
weight_decay = 0.0005
epochs  = 300
root= "E:\\pifujing\\data\\ISIC-2017\\"




if __name__=='__main__':


    dataset = Data(root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
    )
    Net = ResFCN().cuda()
    # path_model = "save_check\\"+"17_super"+".pth"
    # Net.load_state_dict(torch.load(path_model))
    optimizer = torch.optim.SGD(Net.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        Net.train()
        for batch_i, (img, label_p) in enumerate(dataloader):
            start_time = time.time()
            batches_done = batch_i + 1
            img = Variable(img.float()).cuda()
            label_p = label_p.float().cuda()
            pre_label = Net(img)

            Loss = torch.nn.BCELoss()
            loss = Loss(pre_label,label_p)
            end_time = time.time()
            t = end_time - start_time
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("epoch:", epoch, " batch_i:", batch_i, " time:", t, " loss:", loss)
            torch.cuda.empty_cache()
        parapth = "check\\" + str(epoch) + ".pth"
        torch.save(Net.state_dict(), parapth)
