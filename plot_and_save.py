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
from tools import select_quantile
from tqdm import tqdm
import sys

# Loading CPU-Net and support functions
from tools import get_ca, get_tail_slope, inf_train_gen, LambdaLR, weights_init_normal, compute_normalized_tail_slope, normalize_tail_slope, calculate_iou, load_data, num_paramters
from dataset import SplinterDataset, SEQ_LEN, LSPAN, RSPAN
from network import PositionalUNet, RNN, EugiFormer
from accelerate import Accelerator
from collections import OrderedDict
import accelerate
import time
import wandb
import sys

#### Performance Validation and Plot ####
ATN_type = sys.argv[1]
os.environ["WANDB_DISABLED"] = sys.argv[6]
deterministic = sys.argv[9] == "True"
MJD = sys.argv[8] == "True"
ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
BATCH_SIZE = 16
target_real = torch.ones(BATCH_SIZE,1)
target_fake = torch.zeros(BATCH_SIZE,1)
train_loader, test_loader = load_data(BATCH_SIZE, MJD=MJD)

# - Load the trained ATN into the device
if ATN_type in {"PU", "og"}:
    ATN = PositionalUNet()
elif ATN_type == "Eugi":
    sample_A, _, _ = next(iter(train_loader))
    seq_len = sample_A.size(-1)
    h_dim = int(sys.argv[2])
    dropout = float(sys.argv[3])
    head = int(sys.argv[4])
    num_layers = int(sys.argv[5])
    reduce = sys.argv[7] == "True"
    ATN = EugiFormer(num_layers=num_layers, h_dim=h_dim, seq_len=seq_len, head=head, dropout=dropout, reduce=reduce)
else:
    raise IndexError
ATN.to(DEVICE)
# pretrained_dict = torch.load(ATN_type + '_ATN.pt', map_location=DEVICE)
pretrained_dict = torch.load(f"./{ATN_type}/" + ATN_type + '_ATN.pt')
# To switch from multi GPU to single GPU (remove `module.`)
new_state_dict = OrderedDict()
for k, v in pretrained_dict.items():
    name = k[7:] 
    new_state_dict[name] = v
# new_state_dict = pretrained_dict

model_dict = ATN.state_dict()
model_dict.update(new_state_dict) 
print(ATN)
ATN.load_state_dict(new_state_dict)
ATN.eval()
accelerator.init_trackers(project_name="CPU", config={"param":num_paramters(ATN), "model":str(ATN)})

## Making below work would make it a lot faster.
# ATN = accelerator.prepare(ATN)
# pretrained_dict = torch.load(ATN_type+'_ATN.pt')
# model_dict = ATN.state_dict()
# model_dict.update(pretrained_dict) 
# ATN.load_state_dict(pretrained_dict)
# ATN.eval()

with torch.no_grad():
    # - Read a single batch from the test loader, translating it through the ATN
    accelerator.print('Read a single batch from the test loader, translating it through the ATN')
    wf, wf_deconv, _ = next(iter(test_loader))
    wf = wf.to(DEVICE)
    wf_deconv = wf_deconv.to(DEVICE)

    outputs  = ATN(wf_deconv, deterministic=deterministic)
    iwf = 2 # the ith waveform in the batch to plot
    detector_pulse = wf[iwf,0,:].cpu().data.numpy().flatten()
    simulated_pulse = wf_deconv[iwf,0,:].cpu().data.numpy().flatten()
    translated_pulse = outputs[iwf,0,:].cpu().data.numpy().flatten()

    # - Plot simulated pulses, data pulses and translated pulses in the same plot.
    accelerator.print('Plot simulated pulses, data pulses and translated pulses in the same plot.')
    fig = plt.figure(figsize=(25, 7))
    plt.plot(detector_pulse, label="Data Pulse",alpha=0.3, color="magenta", linestyle=":",linewidth = 4)
    plt.plot(simulated_pulse, label="Simulated Pulse",alpha=0.7, color="red", linewidth = 3)
    plt.plot(translated_pulse, label="ATN Output",color="dodgerblue", linewidth = 2)
    plt.axvspan(xmin=300,xmax=358,alpha=0.2,color="grey", label="Preamp Integration")
    plt.axvspan(xmin=358,xmax=800,alpha=0.1,color="cyan",label="RC Discharge")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("Time Sample [ns]")
    plt.ylabel("ADC Counts [a.u.]")
    # ax_main.plot(orwf, label="Data->Siggen",alpha=0.3,color="green", linewidth = 5)
    # plt.gca().get_xaxis().set_visible(False)
    # plt.gca().get_yaxis().set_visible(False)
    plt.legend(loc="upper left")
    plt.xlim(200, 600)
    plt.savefig(f"./{ATN_type}/" + ATN_type+"_ATN.png",dpi=200)
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()

    # - Obtain the critical reconstruction parameters of each waveform by looping through the test dataset
    #     - `ca`: maximal current amplitude
    #     - `ts`: tail slope
    # - Note that this code is slow, mainly because of the current amplitdue calculation
    ts, gan_ts, ca, gan_ca, sim_ca = compute_normalized_tail_slope(ATN, test_loader, DEVICE, deterministic=deterministic, accelerator=accelerator)
    n_ts, n_gan_ts = normalize_tail_slope(ts, gan_ts)
    
    # - Plotting the normalized tail slope
    fig = plt.figure(figsize=(10,8))
    plt.rcParams['font.size'] = 20
    plt.rcParams["figure.figsize"] = (10,8)
    rg = np.linspace(-4,16,50)
    plt.hist(n_ts,bins=rg,histtype="step",linewidth=2,density=False,color="dodgerblue",label="Detector Pulse")
    plt.hist(n_gan_ts,bins=rg,histtype="step",linewidth=2,density=False,color="magenta",label="ATN Output Pulse")
    plt.axvline(x=0,color="deeppink",linewidth=3,label="Simulated Pulse")
    plt.legend()
    plt.ylabel("# of Waveforms/ 0.02 [a.u.]")
    plt.xlabel("Normalized Tail Slope [a.u.]")
    plt.savefig(f"./{ATN_type}/" + ATN_type + "_tailslope.png",dpi=100)
    print("normalized slope_HIoU:")
    slope_HIoU = calculate_iou(ts, gan_ts, rg=rg, normed=True)
    print(slope_HIoU)
    accelerator.log({"slope_HIoU": slope_HIoU})
    print("-------")
    
    # - Plotting the maximal current amplitude
    fig = plt.figure(figsize=(10,8))
    plt.rcParams['font.size'] = 20
    plt.rcParams["figure.figsize"] = (10,8)
    rg = np.linspace(0.05,0.12,50)
    plt.hist(gan_ca,bins=rg,label="ATN Output Pulse",alpha=0.1,color="magenta")
    plt.hist(sim_ca,bins=rg,label="Simulated Pulse",linewidth=2,histtype="step",color="deeppink")
    plt.hist(ca,bins=rg,label="Detector Pulse",histtype="step",linewidth=2,color="dodgerblue")
    plt.xlabel("Current Amplitude [Normalized ADC Count / 100 ns]")
    plt.ylabel("# of Events / 0.001 Current Amplitude")
    plt.legend(loc="upper left")
    # plt.yscale("log")
    plt.savefig(f"./{ATN_type}/"+ATN_type+"_current_amp.png",dpi=200)
    print("current AMP HIOU:")
    amp_hiou = calculate_iou(gan_ca, ca, rg=rg)
    print(amp_hiou)
    accelerator.log({"amp_hiou": amp_hiou})
    