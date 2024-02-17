# # !/bin/bash
# #!/geom/bin/python

# ATN_type : (PU/og/Eugi)
# reduce: is only for Eugi(i.e. Transformers) but tbh it should always be None or True.
# Deterministic: reparamtrization during eval

## train.py args:         ATN_type | epoch | h_dim | dropout | head | num_layers | disable Wandb | reduce | MJD | LRATE |
## plot_and_save.py args: ATN_type         | h_dim | dropout | head | num_layers | disable Wandb | reduce | MJD | Deterministc | 
# accelerate launch train.py PU 3    None None None None True  None  False None  
# python plot_and_save.py    PU      None None None None False None  False True 

# accelerate launch train.py PU 3    None None None None True  None  True None  
# python plot_and_save.py    PU      None None None None False None  True False 

# python generate_w_diffusion.py --epochs 64
# accelerate launch diffusion.py --epochs 32
# python generate_w_diffusion.py --epochs 32

# accelerate launch diffusion.py --epochs 32
# python generate_w_diffusion.py --epochs 32

python generate_w_diffusion.py --epochs 16

## train.py args:         ATN_type | epoch | h_dim | dropout | head | num_layers | disable Wandb | reduce | MJD | LRATE |
# accelerate launch train.py Eugi        3      80        0.0      5        6           True          True   False  0.0001  
## plot_and_save.py args: ATN_type         | h_dim | dropout | head | num_layers | disable Wandb | reduce | MJD | Deterministc | 
# python plot_and_save.py    Eugi               80        0.0      5        6           False         True   False     False