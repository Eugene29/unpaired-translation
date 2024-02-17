'''
This script contains the PositionalUNet network along with 3 candidate discriminators:
* RNN+Attention discriminator
* CNN+PositionalEncoding Discriminator
* Fully Connected Discriminators
we have tested all 3 discriminators, turns out that the RNN+Attention works the best
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math
from dataset import SEQ_LEN
from performer_pytorch import SelfAttention

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=11, padding=5,bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=7, padding=3,bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, attention=False, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
#         if attention:
#             self.Q = Linear(in_channels, out_channels)
#             self.K = Linear(in_channels, out_channels)
#             self.V = Linear(in_channels, out_channels)
    def forward(self, x1, x2, K=None, V=None):
#         if attention:
#             Q = self.Q(x1)
#             K, V = self.K(x2), self.V(x2)
#             print(V.size(-2))
#             A = Q @ K.T / torch.sqrt(V.size(-2)) * V
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # torch.nn.LeakyReLU(),
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, start=0, dropout=0.1, max_len=10000,factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.factor = factor

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)
        self.start = start
    # @torchsnooper.snoop()
    def forward(self, x, need_pos=False):
        x = x + self.factor*self.pe[:,:,self.start:(self.start+x.size(2))]
        x = self.dropout(x)
        if need_pos:
            return x, self.factor*self.pe[:,:,self.start:(self.start+x.size(2))]
        return x
    
class PositionalUNet(nn.Module):
    def __init__(self):
        super(PositionalUNet, self).__init__()
        self.bilinear = True
        
        multi = 40
        
        self.inc = DoubleConv(1, multi)
        self.down1 = Down(multi, multi*2)
        self.down2 = Down(multi*2, multi*4)
        self.down3 = Down(multi*4, multi*8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(multi*8, multi*16 // factor)
        
        self.fc_mean = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        self.fc_var = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        
        self.up1 = Up(multi*16, multi*8 // factor, self.bilinear)
        self.up2 = Up(multi*8, multi*4 // factor, self.bilinear)
        self.up3 = Up(multi*4, multi*2 // factor, self.bilinear)
        self.up4 = Up(multi*2, multi // factor, self.bilinear)
        self.outc = OutConv(multi // factor, 1)
        
        self.pe1 = PositionalEncoding(multi)
        self.pe2 = PositionalEncoding(multi*2)
        self.pe3 = PositionalEncoding(multi*4)
        self.pe4 = PositionalEncoding(multi*8)
        self.pe5 = PositionalEncoding(multi*16//factor)
        self.pe6 = PositionalEncoding(multi*8// factor,start=multi*4)
        self.pe7 = PositionalEncoding(multi*4// factor,start=multi*2)
        self.pe8 = PositionalEncoding(multi*2// factor,start=multi*2)
#         self.pe9 = PositionalEncoding(multi// factor,start=0,factor=1.0)
    
    def reparametrize(self, mu, logvar, deterministic):
        if deterministic:
            return mu
        else:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(mu)
            print("Reparametrization is turned off.")
            # print("uh oh, reparamtrization during evaluation? nuh uh!")
            return eps.mul(std).add_(mu)
    
    # @torchsnooper.snoop()
    def forward(self, x, deterministic=False):        
        x1 = self.pe1(self.inc(x))
        x2 = self.pe2(self.down1(x1))
        x3 = self.pe3(self.down2(x2))
        x4 = self.pe4(self.down3(x3))
        x5 = self.down4(x4)
        x5 = self.pe5(self.reparametrize(self.fc_mean(x5), self.fc_var(x5), deterministic))
        
        x = self.pe6(self.up1(x5, x4))
        x = self.pe7(self.up2(x, x3))
        x = self.pe8(self.up3(x, x2))
        x = self.up4(x, x1)
        output = self.outc(x)
        out = []
        
        # Normalize the output waveforms to interval between [0,1]
        for ibatch in range(output.size(0)):
            out.append(((output[ibatch,0] - output[ibatch,0].min()) / (output[ibatch,0].max() - output[ibatch,0].min())).unsqueeze(0).unsqueeze(0))
        output = torch.cat(out,dim=0)
        return output

class Diffusion_PUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilinear = True
        SEQ_LEN = 800
        multi = 40
        
        self.inc = DoubleConv(1, multi)
        self.down1 = Down(multi, multi*2)
        self.down2 = Down(multi*2, multi*4)
        self.down3 = Down(multi*4, multi*8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(multi*8, multi*16 // factor)
        
        self.fc_mean = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        self.fc_var = torch.nn.Conv1d(multi*16 // factor, multi*16 // factor,1)
        
        self.up1 = Up(multi*16, multi*8 // factor, self.bilinear)
        self.up2 = Up(multi*8, multi*4 // factor, self.bilinear)
        self.up3 = Up(multi*4, multi*2 // factor, self.bilinear)
        self.up4 = Up(multi*2, multi // factor, self.bilinear)
        self.outc = OutConv(multi // factor, 1)
        
        self.pe1 = PositionalEncoding(multi)
        self.pe2 = PositionalEncoding(multi*2)
        self.pe3 = PositionalEncoding(multi*4)
        self.pe4 = PositionalEncoding(multi*8)
        # self.pe5 = PositionalEncoding(multi*16//factor)
        self.pe6 = PositionalEncoding(multi*8 // factor,start=multi*4)
        self.pe7 = PositionalEncoding(multi*4 // factor,start=multi*2)
        self.pe8 = PositionalEncoding(multi*2 // factor,start=multi*2)

        def create_ffnn(o_dim):
            return nn.Sequential(
                # nn.Linear(1, 2*o_dim),
                # nn.GELU(),
                # nn.Linear(2*o_dim, o_dim),                
                nn.Linear(1, o_dim),
            )
        out_dims = [SEQ_LEN*20] * 5
        self.time_ffnns = nn.ModuleList([create_ffnn(o_dim) for o_dim in out_dims])
#         self.pe9 = PositionalEncoding(multi// factor,start=0,factor=1.0)
    
    # def reparametrize(self, mu, logvar, deterministic):
    #     if deterministic:
    #         return mu
    #     else:
    #         std = logvar.mul(0.5).exp_()
    #         eps = torch.randn_like(mu)
    #         print("uh oh, reparamtrization during evaluation? nuh uh!")
    #         return eps.mul(std).add_(mu)
    
    # @torchsnooper.snoop()
    def forward(self, x, t, deterministic=False): 
        t = t[:, None].float()
        x1 = self.pe1(self.inc(x))
        x2 = self.pe2(self.down1(x1))
        x3 = self.pe3(self.down2(x2))
        x4 = self.pe4(self.down3(x3))
        x5 = self.down4(x4)
        # x5 = self.pe5(self.reparametrize(self.fc_mean(x5), self.fc_var(x5), deterministic))
        # x5 = self.pe5(self.reparametrize(self.fc_mean(x5), self.fc_var(x5), deterministic))

        x = self.pe6(self.up1(x5 + self.time_ffnns[0](t).view(*x5.shape), x4))
        x = self.pe7(self.up2(x + self.time_ffnns[1](t).view(*x.shape), x3))
        x = self.pe8(self.up3(x + self.time_ffnns[2](t).view(*x.shape), x2))
        x = self.up4(x + self.time_ffnns[3](t).view(*x.shape), x1)
        output = self.outc(x + self.time_ffnns[4](t).view(*x.shape))
        return output
    
class EugiFormer(nn.Module):
    def __init__(self, num_layers, h_dim, seq_len, head, dropout, reduce=True):
        super().__init__()
        self.reduce = reduce
        self.num_layers = num_layers
        h_dim = h_dim if reduce else seq_len ## switch h_dim to seq_len if reduce is False
        h_dims = [h_dim if reduce else seq_len for _ in range(num_layers)]
        def create_ffnn(input_dim=h_dim, hidden_dim=2 * h_dim, output_dim=h_dim):
            ffnn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
            return ffnn

        self.pos_encoder = PositionalEncoding(h_dim)
        self.preprocess = create_ffnn(input_dim=h_dim)
#         print(sum([p.numel() for p in self.ffnn1.parameters() if p.requires_grad]))
#         self.Q = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(num_layers)])
#         self.K = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(num_layers)])
#         self.V = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(num_layers)])
        self.mhas = nn.ModuleList([nn.MultiheadAttention(embed_dim=h, num_heads=head, dropout=0, batch_first=True) for h in h_dims])
#         self.performers = nn.ModuleList([SelfAttention(dim=h, heads=head, dropout=0, causal=False) for h in h_dims])
        self.ffnns = nn.ModuleList([create_ffnn() for _ in h_dims])
        self.bns1 = nn.ModuleList([nn.BatchNorm1d(num_features=h) for h in h_dims])
        self.bns2 = nn.ModuleList([nn.BatchNorm1d(num_features=h) for h in h_dims])
#         self.lns1 = nn.ModuleList([nn.LayerNorm(h) for h in h_dims])
#         self.lns2 = nn.ModuleList([nn.LayerNorm(h) for h in h_dims])
        self.dropout = nn.Dropout(p=dropout)
        self.postprocess = create_ffnn(output_dim=1)
            
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model = d_model,
#             nhead=head,
#             dim_feedforward=2*d_model,
#             dropout=dropout
#         )
#         self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers)
    def forward(self, x, deterministic=None):
#         Need extra dimension for PE b/c it is applied to each token's embedding. 
#         x = self.dropout(self.preprocess(x.transpose(-2, -1)).transpose(-2, -1)) ## [B, F, T]
        x, pos = self.pos_encoder(x, need_pos=True) # [B, F, T]
        pos = pos.transpose(-2, -1)
        x = self.preprocess(x.transpose(-2, -1)) ## [B, T, F]
#         x = F.dropout(x, p=0.2)
        
        for i in range(self.num_layers):
            A, _ = self.mhas[i](x, x, x, need_weights=False)
#             A = self.performers[i](x)
            x = self.dropout(A) + x
#             x = F.dropout(x, p=0.2)
#             x = self.lns1[i](x)
            x = self.bns1[i](x.transpose(-2, -1)).transpose(-2, -1) ## this is how u use BN1d on 3d tensor (it wants feature dims right after the batch)
            x = self.dropout(self.ffnns[i](x)) + x
#             x = self.lns2[i](x)
#             x = F.dropout(x, p=0.2)
            x = self.bns2[i](x.transpose(-2, -1)).transpose(-2, -1) ## this is how u use BN1d on 3d tensor (it wants feature dims right after the batch)
        x = self.postprocess(x).transpose(-2, -1) # [B, T, F] => [B, 1, T]
        output = x

        ## Normalizing waveorm?
        out = []
        for ibatch in range(output.size(0)):
            out.append(((output[ibatch,0] - output[ibatch,0].min()) / (output[ibatch,0].max() - output[ibatch,0].min())).unsqueeze(0).unsqueeze(0))
        output = torch.cat(out,dim=0)
        return output
    
#The RNN based model:
class RNN(nn.Module):
    def __init__(self,get_attention = False):
        super(RNN, self).__init__()
        
        bidirec = True    #Whether to use a bidirectional RNN
        self.bidirec =bidirec
        feed_in_dim = 128
        self.seg = 1      #Segment waveform to reduce its length. If the original waveform is (2000,1), then segment it with self.seg=5 can reduce its length to (400,5)
        self.emb_dim = 64
        self.emb_tick = 1/1000.0
        self.embedding = nn.Embedding(int(1/self.emb_tick),self.emb_dim)
        self.seq_len = (SEQ_LEN)//self.seg
        if bidirec:
            self.RNNLayer = torch.nn.GRU(input_size = self.emb_dim, hidden_size = feed_in_dim//2,num_layers=1, batch_first=True,bidirectional=True,dropout=0.0)
            feed_in_dim *= 2
        else:
            self.RNNLayer = torch.nn.GRU(input_size = self.emb_dim, hidden_size = feed_in_dim//2,num_layers=1, batch_first=True,bidirectional=False,dropout=0.0)
        self.attention_weight = nn.Linear(feed_in_dim//2, feed_in_dim//2, bias=False)
        self.norm = torch.nn.BatchNorm1d(feed_in_dim//2)
        self.get_attention = get_attention
        
        fc1, fc2 = (feed_in_dim, int(feed_in_dim*0.25))
        do = 0.2
        self.fcnet = nn.Linear(fc1, 1)

    def forward(self, x):
        x = x.view(-1,self.seq_len)
        x = (x - x.min(dim=-1,keepdim=True)[0])/(x.max(dim=-1,keepdim=True)[0] - x.min(dim=-1,keepdim=True)[0])
        x = (x/self.emb_tick).long()
        x = self.embedding(x)
        bsize = x.size(0)
        output, hidden = self.RNNLayer(x)
        if self.bidirec:
            hidden =  hidden[-2:]
            hidden = hidden.transpose(0,1).reshape(bsize,-1)
        else:
            hidden =  hidden[-1]
        
        # Cosine Attention
        inner_product = torch.einsum("ijl,il->ij",output, hidden)
        output_norm = torch.linalg.norm(output,dim=-1)
        hidden_norm = torch.linalg.norm(hidden,dim=-1).unsqueeze(-1).expand(output_norm.size())
        attention_score = torch.softmax(inner_product/(output_norm*hidden_norm),dim=-1) #Softmax over seq_len dimension
        
        if self.get_attention:
            return attention_score
        
        context = torch.sum(attention_score.unsqueeze(-1).expand(*output.size()) * output,dim=1) #Sum over seq_len dimension with attention score multiplied to output
        x = self.fcnet(torch.cat([context,hidden],dim=-1)) #concatenate context vector with last hidden state output

        return torch.sigmoid(x)

class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
    
        do = 0.5
        self.seq_len = LSPAN+RSPAN
        conv1, conv2, conv3, conv4 = (8,16,24,32)
        self.CNNBackbone = nn.Sequential(
            torch.nn.Conv1d(1,conv1,8),
            torch.nn.BatchNorm1d(conv1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            PositionalEncoding(conv1,start=0,dropout=do),
            torch.nn.Conv1d(conv1,conv2,6),
            torch.nn.BatchNorm1d(conv2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            PositionalEncoding(conv2,start=0,dropout=do),
            torch.nn.Conv1d(conv2,conv3,4),
            torch.nn.BatchNorm1d(conv3),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            PositionalEncoding(conv3,start=0,dropout=do),
            torch.nn.Conv1d(conv3,conv4,4),
            torch.nn.BatchNorm1d(conv4),
            torch.nn.LeakyReLU(),
            PositionalEncoding(conv4,start=0,dropout=do),
        )
        self.fcnet = nn.Sequential(
            torch.nn.Linear(2976 , 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(do),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        batch = x.size(0)
        x = self.CNNBackbone(x)
        x = x.view(batch,-1)
        x = self.fcnet(x)
        return x

class FCDiscriminator(nn.Module):
    def __init__(self):
        super(FCDiscriminator, self).__init__()

        do = 0.2
        self.seq_len = LSPAN+RSPAN
        conv1, conv2, conv3, conv4 = (8,16,24,32)

        self.fcnet = nn.Sequential(
            torch.nn.Linear(self.seq_len,512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(do),
            torch.nn.Linear(512, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(do),
            torch.nn.Linear(128, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(do),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        batch = x.size(0)
        x = x.view(-1,self.seq_len)
        x = self.fcnet(x)
        return x