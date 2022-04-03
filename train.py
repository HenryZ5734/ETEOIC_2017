import os
import torch.optim
import logging

from model.net import Net
from loss import Loss
from data.dataloader import dataLoader
from utils.log import init_log
from utils.tensor2image import tensor2image


def train(net, train_iter, loss, d_lambda, optimizer, num_epoch, device, result_path):
    net.to(device)
    net.train()
    for i in range(num_epoch):

        # train
        for X in train_iter:
            optimizer.zero_grad()
            X.to(device)
            X_hat, rate, distortion = net(X)
            l = loss(rate, distortion, d_lambda)
            l.backward()
            optimizer.step()

        # train_log
        if i == 0:
            logging.info(f"num_epoch:{num_epoch}, d_lambda:{d_lambda}, device:{device}")
        if (i+1) % 5 == 0:
            logging.info(f"epoch_{i} loss:{l} rate:{rate} distortion{distortion}")
            tensor2image(X_hat, i+1, result_path)  # tensor2image
        else:
            logging.info(f"epoch_{i} loss:{l}")


def main():
    net = Net(device="cpu", is_train=True)
    train_iter = dataLoader(batch_size=1, root=os.path.join(os.getcwd(), "image", "train"))
    loss = Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)   # 怎么直到net的param是哪些
    num_epoch = 5
    d_lambda = 1000
    result_path = init_log(os.getcwd())
    train(net=net, train_iter=train_iter, loss=loss, d_lambda=d_lambda, optimizer=optimizer, num_epoch=num_epoch,
          device="cpu", result_path = result_path)


if __name__ == "__main__":
    main()
