import torch
import torch.utils.data as data
import os
import warnings
from time import time
from networks.unet import Unet
from networks.dunet import Dunet
from networks.linknet import LinkNet34
from networks.dlinknet import DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.nllinknet import NL34_LinkNet
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

warnings.filterwarnings("ignore")
ROOT = 'dataset/train/'
NAME = 'trainlog_DinkNet34'  # 保存日志名
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))
batchsize = 2

if __name__ == '__main__':
    solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
    dataset = ImageFolder(trainlist, ROOT)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    no_optim = 0
    total_epoch = 300  # 训练轮数
    train_epoch_best_loss = 100.
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)
        print('********', file=mylog)
        print('epoch:', epoch, 'time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/' + NAME + '.pt')
        if no_optim > 6:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/' + NAME + '.pt')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()
