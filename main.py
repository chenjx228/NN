import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import os.path as osp

from nn.data.dataset import MNIST
from nn.data import DataLoader
from nn.optim import MBGD

from utils import parse_data, L2Loss, L2Loss_deriv, accuracy
from models import BPNN, CNN

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--savepath', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--class_num', type=int, default=10)
parser.add_argument('--height', type=int, default=28)
parser.add_argument('--width', type=int, default=28)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=20)
args = parser.parse_args()


def train(epoch, max_epoch, model, dataloader, optimizer, class_num, print_freq=20):
    acc_mean = 0.0
    loss_mean = 0.0
    cnt = 0
    for batch_idx, data in enumerate(dataloader):
        imgs, labels = parse_data(data, class_num)
        batch_size = imgs.shape[0]
        # print(imgs.shape)

        # print(imgs[0])
        outputs = model(imgs)

        # if batch_idx == 0:
        #     print(outputs[0])

        loss_deriv = L2Loss_deriv(outputs, labels)
        optimizer.backward(loss_deriv)
        optimizer.update(imgs)

        loss = L2Loss(outputs, labels)
        loss_mean = (loss * batch_size + cnt * loss_mean) / (cnt + batch_size)
        acc = accuracy(outputs, labels)
        acc_mean = (acc * batch_size + cnt * acc_mean) / (cnt + batch_size)
        cnt += batch_size
        if (batch_idx+1) % print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]  '
                  'Loss {loss:.4f} ({loss_mean:.4f})  '
                  'Acc {acc:.2f} ({acc_mean:.2f})  '
                  'Lr {lr:.6f}  '.format(
                epoch + 1, max_epoch, batch_idx + 1, len(dataloader),
                loss=loss,
                loss_mean=loss_mean,
                acc=acc,
                acc_mean=acc_mean,
                lr=optimizer.lr
                )
            )
    return loss_mean, acc_mean


def test(model, dataloader, class_num):
    outputs_all = None
    labels_all = None
    print('Starting predicting...')
    for batch_idx, data in enumerate(dataloader):
        imgs, labels = parse_data(data, class_num)

        outputs = model(imgs)

        if outputs_all is None and labels_all is None:
            outputs_all = outputs
            labels_all = labels
        else:
            outputs_all = np.vstack([outputs_all, outputs])
            labels_all = np.vstack([labels_all, labels])

    print('Computing accuracy...')
    acc = accuracy(outputs_all, labels_all)

    print('** Results **')
    print('Acc {acc:.2f} '.format(acc=acc))

    return acc


def main():
    print('##### Loading Dataset #####')
    trainset = MNIST(root=args.root)
    trainloader = DataLoader(dataset=trainset, shuffle=True, batch_size=args.batch_size)
    testset = MNIST(root=args.root, mode='Test')
    testloader = DataLoader(dataset=testset, shuffle=False, batch_size=args.batch_size)
    # model = BPNN(args.height*args.width, args.class_num)
    model = CNN(dim_in=1, dim_out=args.class_num, height=args.height, width=args.width)
    optimizer = MBGD(model=model, lr=args.lr)
    # print(id(model), id(optimizer.model))
    # print(id(model.modules), id(optimizer.modules))

    train_log = list()
    test_log = list()
    print('##### Start Training #####')
    for epoch in range(args.max_epoch):
        # print(model.modules[1].weight[0, 0])
        loss, train_acc = train(epoch, args.max_epoch, model, trainloader, optimizer, args.class_num, args.print_freq)
        train_log.append((loss, train_acc))

        if (epoch + 1) % args.eval_freq == 0:
            print('##### Testing in Epoch {epoch} #####'.format(epoch=epoch+1))
            test_acc = test(model, testloader, args.class_num)
            test_log.append(test_acc)

    if not osp.exists(args.savepath):
        os.mkdir(args.savepath)
    with open(osp.join(args.savepath, 'train.txt'), 'w') as fp:
        for item in train_log:
            fp.write(str(item[0]) + ' ' + str(item[1]) + '\r\n')
    with open(osp.join(args.savepath, 'test.txt'), 'w') as fp:
        for item in test_log:
            fp.write(str(item) + '\r\n')

    print('##### Final Testing #####')
    test(model, testloader, args.class_num)


if __name__ == '__main__':
    main()
