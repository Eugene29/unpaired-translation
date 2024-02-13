import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
import argparse
from tools import select_quantile, load_data, num_paramters, get_ca, get_tail_slope, inf_train_gen, LambdaLR, weights_init_normal, compute_normalized_tail_slope, normalize_tail_slope, calculate_iou
from network import Diffusion_PUNet # PositionalUNet, 
import matplotlib.pyplot as plt
from tqdm import tqdm
# from accelerate import set_seed

## Steps in my mind:
# 1. Code diffusion model with Some U-net
# 2. Code stable diffusion model with VAEs and attention.

## https://learn.deeplearning.ai/diffusion-models/lesson/5/training
## Next Todo List:
# Diffusion_PUNet
# Train

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
args = parser.parse_args()

# set_seed(42)
ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = accelerate.Accelerator(split_batches=True, kwargs_handlers=[ddp_kwargs])
parser = argparse.ArgumentParser()
BATCH_SIZE = 16
MJD = False
wandb = False

# if accelerator.is_main_process:
#     accelerator.print(netG_B2A)
#     accelerator.print(f"num parameter of ATN: {num_paramters(netG_B2A)}")
# accelerator.init_trackers(project_name="CPU", config={"param":num_paramters(netG_B2A), "model":str(netG_B2A)})
train_loader, test_loader = load_data(BATCH_SIZE, MJD=MJD, validation_split=0.2)

# if wandb:
#     import wandb
#     acceleartor.
# else:
#     accelerator

beta1 = 1e-4
beta2 = 0.02
timestep = 500
b_t = (beta2 - beta1) * torch.linspace(0, 1, timestep+1, device=accelerator.device) + beta1 ## beta1 -> beta2 in 500 time steps. torch.linspace works as percentage. 
a_t = 1 - b_t ## flipped b_t. (1-beta1) -> (1-beta2) which is decreasing. 
ab_t = torch.cumsum(a_t.log(), dim=0).exp() # a_t is decreasing => a_t.log() is decreasing fast => ab_t is getting close to 0
ab_t[0] = 1

# ## could try something called ddim later? 10-50x faster.
# def denoise_add_noise(x, t, pred_noise, z=None):
#     if z is None:
#         z = torch.randn_like(x)
#     noise = b_t.sqrt()[t] * z ## b_t.sqrt() is increasing percentage
#     mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt() ## subtracts some portion of pred_noise and then scaled by percentage decreasing by timestep t
#     return noise + mean ## adds noise to avoid "collapsing" but not sure if its needed

# for epoch in range(epochs):

pbar = tqdm(range(args.epochs))

## perturbs 0-t timestep noise at once. The math turns out that we only have to do this once. It would be good practice to derive it.
def forward_diffusion(x, t, noise):
    # print(a_t.device)
    # print(ab_t.device)
    # print(b_t.device)
    return ab_t.sqrt()[t, None, None] * x + (1 - ab_t.sqrt()[t, None, None]) * noise

m = Diffusion_PUNet()
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
# m, optimizer, train_loader, test_loader, b_t, a_t, ab_t = accelerator.prepare(m, optimizer, train_loader, test_loader, b_t, a_t, ab_t)
m, optimizer, train_loader, test_loader = accelerator.prepare(m, optimizer, train_loader, test_loader)
accelerator.print(f"Total num of parameters: {sum([p.numel() for p in m.parameters()])}")
# b_t = b_t.to(accelerator.device)
# a_t = a_t.to(accelerator.device)
# ab_t = ab_t.to(accelerator.device)

for epoch in pbar:
    total_loss = 0
    for i, (real_A, real_B, _) in enumerate(train_loader):
        ## real_A = detector, real_B = simulated
        optimizer.zero_grad()
        noise = torch.randn_like(real_B)
        t = torch.randint(1, timestep+1, (BATCH_SIZE//accelerator.state.num_processes,), device=accelerator.device)
        diffused = forward_diffusion(real_B, t, noise)
        out = m(diffused, t/timestep)
        loss = F.mse_loss(out, noise)
        total_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
    accelerator.print(f"loss: {total_loss / len(train_loader)}")
    total_loss = 0
    for i, (real_A, real_B, _) in enumerate(test_loader):
        ## real_A = detector, real_B = simulated
        with torch.no_grad():
            noise = torch.randn_like(real_B)
            t = torch.randint(1, timestep+1, (BATCH_SIZE//accelerator.state.num_processes,), device=accelerator.device)
            diffused = forward_diffusion(real_B, t, noise)
            out = m(diffused, t/timestep)
            loss = F.mse_loss(out, noise)
            total_loss += loss
    # accelerator.print(f"loss: {total_loss / len(train_loader)}") ## seems like we don't need gather_for_metrics?
    accelerator.print(f"loss: {accelerator.gather_for_metrics(total_loss).sum() / len(test_loader)}")

## print last diffused image and restored image
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(range(800), real_B[2].detach().cpu().squeeze())
plt.plot(range(800), diffused[2].detach().cpu().squeeze())
plt.plot(range(800), out[2].detach().cpu().squeeze())
plt.savefig("testing.png", dpi = 300)
# Show the plot
plt.show()

if hasattr(m, "module"):
    torch.save(m.module.get_state_dict(), "pre_trained_diffusion.pth")
else:
    torch.save(m.get_state_dict(), "pre_trained_diffusion.pth")

## this function is for after having a trained model
# def sample_ddpm(n_sample, save_rate=20):
#     samples = torch.randn()

