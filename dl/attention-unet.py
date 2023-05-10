import torch
import torch.nn as nn
import torch.utils.data as tdata
from torch.utils.data import Dataset, DataLoader
import utils
import os, tqdm, cv2, random, decimal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    """
    Arch: (Conv3x3->BN->ReLU)x2
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int=3):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=1, bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        return x


class UpConvBlock(nn.Module):
    '''
    Arch: Upsample->Conv3x3->BN->ReLU
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self, feat_g:int, feat_l:int, feat_int:int) -> None:
        super(AttentionGate, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(feat_g, feat_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(feat_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(feat_l, feat_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(feat_int)
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(feat_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g_out = self.w_g(g)
        x_out = self.w_x(x)
        y = self.relu(g_out+x_out)
        psi = self.psi(y)
        return psi*x

class AttentionUNet(nn.Module):
    def __init__(self, in_channel:int=3, out_channel:int=1):
        super(AttentionUNet, self).__init__() 
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = ConvBlock(in_channel, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)
        
        self.up5 = UpConvBlock(1024, 512)
        self.att5 = AttentionGate(512, 512, 256)
        self.upconv5 = ConvBlock(1024, 512)
        
        self.up4 = UpConvBlock(512, 256)
        self.att4 = AttentionGate(256, 256, 128)
        self.upconv4 = ConvBlock(512, 256)
        
        self.up3 = UpConvBlock(256, 128)
        self.att3 = AttentionGate(128, 128, 64)
        self.upconv3 = ConvBlock(256, 128)
        
        self.up2 = UpConvBlock(128, 64)
        self.att2 = AttentionGate(64, 64, 32)
        self.upconv2 = ConvBlock(128, 64)
        
        self.conv_1x1 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # encoder
        x1 = self.conv1(x) #64
        x2 = self.conv2(self.maxpool(x1)) #128
        x3 = self.conv3(self.maxpool(x2)) #256
        x4 = self.conv4(self.maxpool(x3)) #512
        x5 = self.conv5(self.maxpool(x4)) #1024
        
        # decoder + concat
        d5 = self.up5(x5) #1024 -> 512
        x4 = self.att5(g=d5, x=x4) #1024 -> 512
        d5 = torch.concat((x4, d5), dim=1) #512+512 -> 1024
        d5 = self.upconv5(d5) #1024 -> 512
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.concat((x3, d4), dim=1)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.concat((x2, d3), dim=1)
        d3 = self.upconv3(d3)
        
        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.concat((x1, d2), dim=1)
        d2 = self.upconv2(d2)
        
        d1 = self.conv_1x1(d2)
        
        return d1
    
class BrainMRIDataset(Dataset):
    def __init__(self, root_path:str, transforms=None):
        end_len, end_mask_len = 4, 9
        data_paths=[]
        path = os.path.join(root_path, "datasets/lgg-seg/lgg-mri-segmentation/kaggle_3m")
        for dir in os.listdir(path):
            dir_path = os.path.join(path, dir)
            if(os.path.isdir(dir_path)):
                for fnames in os.listdir(dir_path):
                    im_path = os.path.join(dir_path, fnames)
                    data_paths.append([dir, im_path])
            else:
                print(f"[INFO] Not a dir: {dir_path}")
        df = pd.DataFrame(data_paths, columns=["dir_name", "image_path"])
        df_imgs = df[~df["image_path"].str.contains("mask")]
        df_msks = df[df["image_path"].str.contains("mask")]

        imgs = sorted(df_imgs["image_path"].values, key=lambda x: int(x[len(x)-end_len-1: -end_len]))
        msks = sorted(df_msks["image_path"].values, key=lambda x: int(x[len(x)-end_mask_len-1: -end_mask_len]))

        df = pd.DataFrame({"patient":df_imgs.dir_name.values,
                           "image_path":imgs, "mask_path":msks})
        df["diagnosis"] = df["mask_path"].apply(lambda x : self.pos_neg_diagnosis(x))
        # cached data frame:
        self.df = df
        self.transform = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = np.array(cv2.imread(str(self.df.iloc[idx, 1])))
        image = image.transpose(2,0,1)
        mask = np.array(cv2.imread(str(self.df.iloc[idx, 2]), 0))
        mask = np.expand_dims(mask, axis=0)
        if self.transform != None:
            augmented = self.transform(image=image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

    @staticmethod
    def pos_neg_diagnosis(mask_path):
        val = np.max(cv2.imread(mask_path))
        return 1 if val > 0 else 0
    
def show_aug(inputs, nrows=5, ncols=5, norm=False):
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0., hspace=0.)
    
    if len(inputs) > 25: inputs = inputs[:25]
    i_ = 0
    for idx in range(len(inputs)):
        # normalization
        if norm:           
            img = inputs[idx].numpy()#.transpose(1,2,0)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225] 
            img = (img*std+mean).astype(np.float32)
        else:
            img = inputs[idx].numpy().astype(np.float32)
            img = img[0,:,:]
        
        plt.subplot(nrows, ncols, i_+1)
        plt.imshow(img); 
        plt.axis('off')
        i_ += 1
    return plt.show()

# make datasets: train_set, val_set, and test_set
dset = BrainMRIDataset(utils.parent_path(os.getcwd()))
num_train = int(dset.__len__()*0.9)
dset, val_set = tdata.random_split(dset, [num_train, dset.__len__()-num_train])
num_train = int(dset.__len__()*0.8)
train_set, test_set = tdata.random_split(dset, [num_train, dset.__len__()-num_train])

# make dataloaders
batch_size = 2**4
train_loader = DataLoader(train_set, batch_size, num_workers=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size, num_workers=2, shuffle=True)
test_loader = DataLoader(test_set, batch_size, num_workers=2, shuffle=True)

images, masks = next(iter(train_loader))
show_aug(images, 4, 4)
show_aug(masks, 4, 4, norm=False)

model = AttentionUNet(3, 1).to(utils.device())
data = torch.randn(1, 3, 512, 512).to(utils.device())
preds = model(data)

print(preds.shape)
print(preds)