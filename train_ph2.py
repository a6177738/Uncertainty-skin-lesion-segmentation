import torch
from model import ResFCN
import time
from data_train_ph2 import Data
from torch.autograd import Variable
from transform import *

batch_size = 16
num_worker = 4
lr = 4e-3
momentum = 0.9
weight_decay = 0.0005
epochs  = 300
root= "E:\\pifujing\\data\\PH2Dataset\\"




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
        for batch_i, (img, label_p,label_b) in enumerate(dataloader):
            start_time = time.time()
            batches_done = batch_i + 1
            img = Variable(img.float()).cuda()
            label_p = label_p.cuda()
            label_b = label_b.cuda()
            pre_label = Net(img)

            sum_p = torch.sum(label_p)  #前景像素数量
            sum_b = torch.sum(label_b) #背景像素数量
            p = pre_label * label_p  # label_p是可靠前景为1，其他位置忽略
            loss_p = torch.sum(label_p - p) / sum_p
            b = pre_label * label_b  # label_b是可靠背景为1，预测应为0，预测值即为损失，其他位置忽略
            loss_b = torch.sum(b) / sum_b

            loss = loss_p+loss_b
            end_time = time.time()
            t = end_time - start_time
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("epoch:", epoch, " batch_i:", batch_i, " time:", t, " loss:", loss)
            torch.cuda.empty_cache()
        parapth = "check\\" + str(epoch) + ".pth"
        torch.save(Net.state_dict(), parapth)
