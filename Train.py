import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging

import matplotlib.pyplot as plt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

alpha = 1.5
beta = 2

def test(model, path, dataset):
    
    global val_loss                                                                     ###### updated #######
    loss_P2_rec = AvgMeter()
    loss_P1_rec = AvgMeter()
    loss_p1p2_rec = AvgMeter()
    loss_total_rec = AvgMeter()
   

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gts, name = test_loader.load_data()                 #gt
        gt = np.asarray(gts, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        
        
        ###### calculate loss & append it to global val_loss #######

        loss_P1 = structure_loss(res, gts)
        loss_P2 = structure_loss(res1, gts)
        loss_p1p2 = sklearn.metrics.mean_squared_error(loss_P1, loss_P2)
        loss = loss_P1 + alpha*loss_P2 + beta*loss_p1p2
        
        loss_P2_rec.update(loss_P2.data, opt.batchsize)
        loss_P1_rec.update(loss_P1.data, opt.batchsize)
        loss_p1p2_rec.update(loss_p1p2.data, opt.batchsize)
        loss_total_rec.update(loss.data, opt.batchsize)
        
        val_loss['val_loss_p1'].append(loss_P1_rec)
        val_loss['val_loss_p2'].append(loss_P2_rec)
        val_loss['val_loss_p1p2'].append(loss_p1p2_rec)
        val_loss['val_loss_total'].append(loss_total_rec)
        
        # eval Dice
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1



def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    global train_loss

    size_rates = [0.75, 1, 1.25] 
    loss_P2_record = AvgMeter()
    loss_P1_record = AvgMeter()
    loss_p1p2_record = AvgMeter()
    loss_total_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2= model(images)
            # ---- loss function ----
            alpha = 1.5
            beta = 2
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_p1p2 = sklearn.metrics.mean_squared_error(loss_P1, loss_P2)
            loss = loss_P1 + alpha*loss_P2 + beta*loss_p1p2
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
                loss_P1_record.update(loss_P1.data, opt.batchsize)
                loss_p1p2_record.update(loss_p1p2.data, opt.batchsize)
                loss_total_record.update(loss.data, opt.batchsize)
          #  train_loss += loss.item()*data.size(0)                                       ############ updated #############
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
            train_loss['train_loss_p1'].append(loss_P1_record)
            train_loss['train_loss_p2'].append(loss_P2_record)
            train_loss['train_loss_p1p2'].append(loss_p1p2_record)
            train_loss['train_loss_total'].append(loss_total_record)
            
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    global dict_plot
   
    test1path = './dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['Val']:   #'CVC-300','GlaS', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB',          ####### updated ########
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)                                                                                    ## validation steps ##
        meandiceA = test(model, test_path, 'TestA' )                                                                          ## test ##
        print('Mean dice score - TestA data: ', meandiceA)                                                                        
        meandiceB = test(model, test_path, 'TestB' )                                                                          ## test ##
        print('Mean dice score - TestB data: ', meandiceB)
        dict_plot['TestA'].append(meandiceA)
        dict_plot['TestB'].append(meandiceB)
        if meandiceA > best:
            best = meandiceA
            torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))
       
    
    
def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    path = f'/content/drive/MyDrive/BTP/{filename}' #/my_checkpoint.pth.tar'
    print("=>Saving Checkpoint")
    torch.save(state, path)

def load_checkpoint(checkpoint):
    print("=>Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################


#############  Train - meandice score ###########
    
# def plot_train_dice(dict_plot=None, name = None):
#     color = ['red', 'lawngreen', 'blue'] #'lime', 'gold', 'm', 'plum', 'blue'
#     line = ['-', "--"]
#     for i in range(len(name)):
#         plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
#         #### transfuse and axhline are just to add horizontal line.. nothing to do with chart data ####
#         ## 'GlaS': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83
#         transfuse = {'Val': 0.902, 'TestA': 0.83, 'TestB': 0.773}   #'CVC-300' 
#         plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
#     plt.xlabel("epoch")
#     plt.ylabel("dice")
#     plt.title('Training - meandice score vs epochs')
#     plt.legend()
#     plt.savefig('train_meandice-epoch.png')


#############  Validation - meandice score ###########
    
def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'blue'] #'lime', 'gold', 'm', 'plum', 'blue'
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        #### transfuse and axhline are just to add horizontal line.. nothing to do with chart data ####
        ## 'GlaS': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83
        transfuse = {'Val': 0.902, 'TestA': 0.83, 'TestB': 0.773}   #'CVC-300' 
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Validation - meandice score vs epochs')  #Train
    plt.legend()
    plt.savefig('val_meandice-epoch.png')
    # plt.show()
 
    #############  Train - loss curve ###########
    
def plot_train_loss(train_loss=None, loss_name = None):
    color = ['red', 'lawngreen', 'blue'] #'lime', 'gold', 'm', 'plum', 'blue'
    line = ['-', "--"]
    for i in range(len(loss_name)):
        plt.plot(train_loss[loss_name[i]], label=loss_name[i], color=color[i], linestyle=line[(i + 1) % 2])
        #### transfuse and axhline are just to add horizontal line.. nothing to do with chart data ####
        ## 'GlaS': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83
        transfuse = {'train_loss_p1': 0.902, 'train_loss_p2': 0.83, 'train_loss_p1p2': 0.773, 'train_loss_total': 0.70}   #'CVC-300' 
        plt.axhline(y=transfuse[loss_name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Training - loss vs epochs')
    plt.legend()
    plt.savefig('train_loss-epoch.png')

 
    #############    Validation - loss curve ###########
    
def plot_val_loss(val_loss=None, valloss_name = None):
    color = ['red', 'lawngreen', 'blue'] #'lime', 'gold', 'm', 'plum', 'blue'
    line = ['-', "--"]
    for i in range(len(valloss_name)):
        plt.plot(val_loss[valloss_name[i]], label=valloss_name[i], color=color[i], linestyle=line[(i + 1) % 2])
        #### transfuse and axhline are just to add horizontal line.. nothing to do with chart data ####
        ## 'GlaS': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83
        transfuse = {'ValA': 0.902, 'TestA': 0.83, 'TestB': 0.773}   #'CVC-300' 
        plt.axhline(y=transfuse[valloss_name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Validation - loss vs epochs')
    plt.legend()
    plt.savefig('val_loss-epoch.png')
    

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
    
if __name__ == '__main__':
    # 'GlaS':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]
    dict_plot = {'Val':[], 'TestA':[], 'TestB':[]}  #'CVC-300'
    #'GlaS', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test'
    name = ['Val', 'TestA', 'TestB']   #'CVC-300'
    
    train_loss = {'train_loss_p1':[], 'train_loss_p2':[], 'train_loss_p1p2':[],  'train_loss_total':[]}
    loss_name = {'train_loss_p1', 'train_loss_p2', 'train_loss_p1p2', 'train_loss_total'}
    
    val_loss = {'val_loss_p1':[], 'val_loss_p2':[], 'val_loss_p1p2':[],  'val_loss_total':[]}
    valloss_name = {'val_loss_p1', 'val_loss_p2', 'val_loss_p1p2', 'val_loss_total'}
    ##################model_name#############################
    model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PolypPVT().cuda()
    load_model = True

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    if load_model:
        pth = f'/content/drive/MyDrive/BTP/my_checkpoint.pth.tar' 
        if os.path.isfile(pth) == True:
            load_checkpoint(torch.load(pth))       #"my_checkpoint.pth.tar"

    for epoch in range(1, opt.epoch):
        if (epoch % 2) == 0:
           checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
           save_checkpoint(checkpoint)
            
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
        plot_train(dict_plot, name)                           # validation--> meandice score vs epochs
        plot_train_loss(train_loss, loss_name)                # training --> loss vs epochs
        plot_val_loss(val_loss, valloss_name)                 # validation -> loss vs epochs
         
    
    # plot the eval.png in the training stage
    plot_train(dict_plot, name)
    plot_train_loss(train_loss, loss_name)
    plot_val_loss(val_loss, valloss_name)
