import  torch
from torch import nn
import numpy as np
from model import Generator,Discriminator
import argparse
from data import TrainDataset,ValDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from loss import GeneratorLoss
from ssim import Ssim
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file,map_location='cpu')  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            if v.shape==model_dict[k].shape:
                state_dict[k] = v
            else:
                print(k,'shape dismatch')
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict



if __name__ == '__main__':

    train_dir = r'E:\study\SR\data\train'
    val_dir = r'E:\study\SR\data\val'

    opt = parser.parse_args()

    gen_train = TrainDataset(train_dir,opt.crop_size,opt.upscale_factor)
    loader_train = DataLoader(gen_train,num_workers=4,batch_size=32,shuffle=True)
    gen_val = ValDataset(val_dir,opt.crop_size,opt.upscale_factor)
    loader_val = DataLoader(gen_val,num_workers=1,batch_size=1,shuffle=False)


    netG = Generator(opt.upscale_factor)
    netG = transfer_model("model_data/netG_epoch_6_29.933033_0.777779.pth", netG).cuda()
    netD = Discriminator()
    netD = transfer_model("model_data/netD_epoch_6_29.933033_0.777779.pth", netD).cuda()

    optimizerG = optim.Adam(netG.parameters(),lr=1e-4)
    shedulerG = StepLR(optimizerG, step_size=1, gamma=0.6)
    optimizerD = optim.Adam(netD.parameters(),lr=1e-4)
    shedulerD = StepLR(optimizerD, step_size=1, gamma=0.6)
    g_loss_cal = GeneratorLoss()


    for iter in range(opt.num_epochs):
        netG.train()
        netD.train()
        g_loss_total = 0; d_loss_total = 0
        # pbar = tqdm(total=len(loader_train)//100,desc=f'Epoch {i + 1}/{opt.num_epochs}',postfix=dict,mininterval=0.3)
        pbar = tqdm(loader_train)
        i = 0
        for lr_image,hr_image in pbar:
            i += 1
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
            fake_image = netG(lr_image)

            # D训练过程
            optimizerD.zero_grad()
            real = netD(hr_image).mean()
            fake = netD(fake_image).mean()
            d_loss = 1 - real + fake
            d_loss_total += d_loss.item()
            d_loss.backward(retain_graph=True)
            optimizerD.step()
            #G训练过程
            optimizerG.zero_grad()
            fake_image = netG(lr_image)
            fake = netD(fake_image).mean()
            g_loss = g_loss_cal(fake,fake_image,hr_image)
            g_loss_total += g_loss.item()
            g_loss.backward()
            optimizerG.step()

            pbar.set_description(
                desc='Iter:%d->[converting LR images to SR images] g_loss: %.6f , d_loss: %.6f' % (iter,
                    g_loss_total / (i) , d_loss_total / (i)))
        shedulerG.step()
        shedulerD.step()

        netG.eval()
        netD.eval()
        g_loss_total = 0
        d_loss_total = 0
        # pbar = tqdm(total=len(loader_val) // 100, desc=f'Epoch {i + 1}/{opt.num_epochs}', postfix=dict,
        #             mininterval=0.3)
        val_bar = tqdm(loader_val)
        # for i, batch in enumerate(loader_val):
        q = 0; mse = 0; psnr = 0; ssim = 0;
        for lr_image,hr_image in val_bar:
            with torch.no_grad():
                q += 1
                lr_image = lr_image.cuda()
                hr_image = hr_image.cuda()
                fake_image = netG(lr_image)

                batch_mse = ((fake_image - hr_image) ** 2).data.mean().cpu().numpy()
                batch_psnr = 10 * np.log10((hr_image.cpu().max() ** 2) / batch_mse)
                mse += batch_mse
                psnr += batch_psnr

                batch_ssim = Ssim(fake_image, hr_image)
                ssim += batch_ssim
            val_bar.set_description(
                desc='Iter:%d->[converting LR images to SR images] PSNR: %.6f dB SSIM: %.6f' % (iter,
                    psnr/(q), ssim/(q)))
        torch.save(netG.state_dict(), 'logs/netG_epoch_%d_%.6f_%.6f.pth' % (iter,psnr/(q),ssim/(q)))
        torch.save(netD.state_dict(), 'logs/netD_epoch_%d_%.6f_%.6f.pth' % (iter,psnr/(q),ssim/(q)))












