import numpy as np
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torchsnooper
from scipy import signal
from torch import nn, einsum
import matplotlib.gridspec as gridspec
import itertools
from tools import select_quantile, load_data,num_paramters
from tqdm import tqdm

# Loading CPU-Net and support functions
from tools import get_ca, get_tail_slope, inf_train_gen, LambdaLR, weights_init_normal, compute_normalized_tail_slope, normalize_tail_slope, calculate_iou
from dataset import SplinterDataset, SEQ_LEN, LSPAN, RSPAN
from network import PositionalUNet, RNN, EugiFormer
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import OrderedDict
import accelerate
import time
import wandb
import sys
import os

#### TODO LIST:
# There is a hidden unused parameter somwhere. Where would it be? 
# immigrate from sys.argv to argsparse
set_seed(42222)
os.environ["WANDB_DISABLED"] = sys.argv[7]
## Accelerator
ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb", split_batches=True)

# # ## U-Net Training
# # - Define global parameter for the training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# BATCH_SIZE = 4
BATCH_SIZE = 16
# ITERS = 3001
ITERS = 1
DECAY = 500
LRATE = 1e-3
target_real = torch.ones(BATCH_SIZE,1)
target_fake = torch.zeros(BATCH_SIZE,1)

## Immigrate to argsparse later
## sys input
if accelerator.is_main_process:
    if len(sys.argv) < 3 and sys.argv[1] not in {"PU", "Eugi"}:
        raise ValueError("Enter args => \n model_type {PU, Eugi}, \n num_epochs")
ATN_type = sys.argv[1]
num_epoch = int(sys.argv[2])
MJD = sys.argv[9] == "True"
accelerator.print(f"chosen ATN_type: {ATN_type}")

# # - Create infinite train generator. This generator can be called an infinite amount of time to draw from training dataset (with repetition)
train_loader, test_loader = load_data(BATCH_SIZE, MJD=MJD)
# print(next(iter(test_loader))[0].shape) # 16, 1, 800
# print(len(test_loader))
# raise ValueError

# - Create network structures and feed them into the DEVICE defined above
#     - A: Detector Pulses
#     - B: Simulated Pulses
#     - BtoA: Ad-hoc Translation Network (Simulation to Data)
#     - AtoB: Inverse Ad-hoc Translation Network (Data to Simulation)

if ATN_type == "PU":
    netG_A2B = PositionalUNet()
    netG_B2A = PositionalUNet()
elif ATN_type == "Eugi":
    sample_A, _, _ = next(iter(train_loader))
    seq_len = sample_A.size(-1)
    h_dim = int(sys.argv[3])
    dropout = float(sys.argv[4])
    head = int(sys.argv[5])
    num_layers = int(sys.argv[6])
    reduce = sys.argv[8] == "True"
    LRATE = float(sys.argv[10])
    netG_A2B = EugiFormer(num_layers=num_layers, h_dim=h_dim, seq_len=seq_len, head=head, dropout=dropout, reduce=reduce)
    netG_B2A = EugiFormer(num_layers=num_layers, h_dim=h_dim, seq_len=seq_len, head=head, dropout=dropout, reduce=reduce)
netD_A = RNN().apply(weights_init_normal)
netD_B = RNN().apply(weights_init_normal)

if accelerator.is_main_process:
    accelerator.print(netG_B2A)
    accelerator.print(f"num parameter of ATN: {num_paramters(netG_B2A)}")
accelerator.init_trackers(project_name="CPU", config={"param":num_paramters(netG_B2A), "model":str(netG_B2A)})
    
# - Create loss function and set up optimizer
#     - BCELoss for discriminator
#     - WFDist is a special L1loss where additional weight is added to the rising and falling edge of the waveform
class WFDist(nn.Module):
    '''
    Waveform Distance, this is a special type of L1 loss which gives more weight to the
    rising and falling edge of each pulse
    '''
    def __init__(self, dev):
        super(WFDist, self).__init__()
        self.dev = dev
        self.criterion = nn.L1Loss().to(dev)
        self.weight = torch.tensor([2.0]*LSPAN+[10.0]*150+[5.0]*(RSPAN-150)).to(dev)
    
    def forward(self, x1, x2):
        loss_out = torch.tensor(0.0, dtype=torch.float32).to(self.dev)
        for i in range(x1.size(0)):
            loss_out = loss_out + self.criterion(x1[i].view(-1)*self.weight, x2[i].view(-1)*self.weight) # /self.weight.sum()
        return loss_out/x1.size(0)

# # In[13]:
#### MODEL TRAINING ####
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=LRATE, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=LRATE, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=LRATE, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(ITERS, 0, DECAY).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(ITERS, 0, DECAY).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(ITERS, 0, DECAY).step)

criterion_GAN = nn.BCELoss()#.to(DEVICE)

netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, train_loader, test_loader, criterion_GAN = accelerator.prepare(netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B, lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, train_loader, test_loader, criterion_GAN)

dev = accelerator.device
target_real, target_fake = target_real.to(dev), target_fake.to(dev)

criterion_cycle = WFDist(dev)#.to(DEVICE)
criterion_identity = WFDist(dev)#.to(DEVICE)

## train_loader size is 1155
## val_loader size is 495
netG_A2B.train()
netG_B2A.train()
start = time.time()
train_end = len(train_loader)
test_end = len(test_loader)
for epoch in range(num_epoch):
# for iteration in tqdm(range(ITERS)):
    #########################
    # A: DetectorPulses
    # B: Simulated Pulses
    #########################
    for i, (real_A, real_B, _) in enumerate(train_loader):
#         real_A, real_B = data
    #     real_A = real_A.to(DEVICE).float()
    #     real_B = real_B.to(DEVICE).float()
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake,target_real)

#         # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_cycle_ABA + loss_cycle_BAB + loss_GAN_A2B + loss_GAN_B2A
        accelerator.backward(loss_G)
        # torch.nn.utils.clip_grad_norm_(netG_B2A.parameters(), 1)    
        # torch.nn.utils.clip_grad_norm_(netG_A2B.parameters(), 1)    
        optimizer_G.step()
        ###### Discriminator A (Detector Pulses) ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = loss_D_real + loss_D_fake
        accelerator.backward(loss_D_A)

        optimizer_D_A.step()
        ###### Discriminator B (Simulated Pulses) ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = loss_D_real + loss_D_fake
        accelerator.backward(loss_D_B)
        
        optimizer_D_B.step()
        percentage_done = (epoch * train_end + i) / (train_end * num_epoch) * 100
        accelerator.print(f"[Training {percentage_done:.1f}% Done] Epoch: {epoch} \n Total Losses: Total: {loss_G + loss_D_A + loss_D_B:.4f} loss_G: {loss_G:.4f} loss_D_A: {loss_D_A:.4f} loss_D_B: {loss_D_B:.4f}")
        accelerator.log({
            "loss_Total": loss_G + loss_D_A + loss_D_B,
            "loss_G": loss_G,
            "loss_D_A": loss_D_A,
            "loss_D_B": loss_D_B
        })
    with torch.no_grad():
        accelerator.print("TORCH NO GRAD SKRRRRT")
        ## test loss to check overfitting
        loss_G = 0
        loss_D_A = 0 
        loss_D_B = 0
        for i, (real_A, real_B, _) in enumerate(test_loader):
            ###### Generators A2B and B2A ######
            # Identity loss
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake,target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_G + loss_identity_A + loss_identity_B + loss_cycle_ABA + loss_cycle_BAB + loss_GAN_A2B + loss_GAN_B2A

            ###### Discriminator A (Detector Pulses) ######

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = loss_D_A + loss_D_real + loss_D_fake

            ###### Discriminator B (Simulated Pulses) ######

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = loss_D_B + loss_D_real + loss_D_fake

            percentage_done = i / test_end * 100
            accelerator.print(f"[Testing {percentage_done:.1f}% Done] Epoch: {epoch}")
        accelerator.print({
            "TEST mean loss_Total": (loss_G.item() + loss_D_A.item() + loss_D_B.item()) / test_end,
            "TEST mean loss_G": loss_G.item() / test_end,
            "TEST mean loss_D_A": loss_D_A.item() / test_end,
            "TEST mean loss_D_B": loss_D_B.item() / test_end
        })
        accelerator.log({
            "TEST mean loss_Total": (loss_G + loss_D_A + loss_D_B) / test_end,
            "TEST mean loss_G": loss_G / test_end,
            "TEST mean loss_D_A": loss_D_A / test_end,
            "TEST mean loss_D_B": loss_D_B / test_end
        })
        ## the following code gives inaccurate HIOU, probably due to some interaction with accelerator? 
#         ts, gan_ts, ca, gan_ca, sim_ca = compute_normalized_tail_slope(netG_B2A, test_loader, DEVICE, accelerator)
#         n_ts, n_gan_ts = normalize_tail_slope(ts, gan_ts)
#         accelerator.log({"HIoU - normalized tail slope": calculate_iou(ts, gan_ts, rg=np.linspace(-4,16,50), normed=True)})
#         accelerator.log({"HIoU - maximal current amplitude": calculate_iou(gan_ca, sim_ca, rg=np.linspace(0.05,0.12,50))})
#-----------------------------------------------
end = time.time()
accelerator.print(f"Time taken to train: {end - start}")
accelerator.log({"total_train_time": end-start})

# - Save trained ATN and IATN
torch.save(netG_B2A.state_dict(), f"./{ATN_type}/" + ATN_type + '_ATN.pt')
torch.save(netG_A2B.state_dict(), f"./{ATN_type}/" + ATN_type + '_IATN.pt')
###END####

## Alert the training is done + wrap up wandb
accelerator.end_training()