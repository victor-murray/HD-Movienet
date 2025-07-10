'''
HD-Movienet 3D version with 4 motion states v1.0 2024/08/08
Input: x5d (NPY file) calculated using UTE acq. time = 2 min (see DOI 10.1088/1361-6560/acace6 for more details)
Output: Movienet 4D (MAT file)
# # # # # # # # # # # # # # # # # # # # 
GPU Example:
CUDA_VISIBLE_DEVICES=1 python HDMovienet_3D_4motionstates.py  myx5d.npy  4Dmrioutput.mat
CPU Example:
CUDA_VISIBLE_DEVICES=, python HDMovienet_3D_4motionstates.py  myx5d.npy  4Dmrioutput.mat
# # # # # # # # # # # # # # # # # # # # 
Please, use this citation:
Murray V, Wu C, Otazo R. High-definition motion-resolved MRI 
using 3D radial kooshball acquisition and deep learning 
spatial-temporal 4D reconstruction. Phys Med Biol. 2025 
Jun 17;70(12). doi: 10.1088/1361-6560/ade195. PMID: 40472864.
'''

#######################################################################
name2load = 'HDMovienet_3D_4motionstates.pt' # 
#
#######################################################################
import torch         #PyTorch
import numpy as np   #NumPy
import os

#######################################################################
NZvolume = 8 # Number of slices to create a 3D block
print('Using volumen with z = {} slices'.format(NZvolume))

#######################################################################
# # # check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Using CPU ...')
else:
    print('CUDA is available!  Using GPU ...')

#######################################################################
def convertInputMAT256(y):
    y = torch.transpose(y, 3, 4)
    y = torch.reshape(y, (y.size(dim=0), y.size(dim=1), y.size(dim=2), -1))
    y = torch.permute(y, (2, 0, 1, 3))
    return y

#######################################################################
def variableforconv3D(y,NZvolume):
    y = torch.permute(y, (1, 2, 3, 0))
    y = torch.reshape(y, (y.size(dim=0), y.size(dim=1), y.size(dim=2), -1, NZvolume))
    y = torch.permute(y, (3, 2, 4, 0, 1))
    return y

#######################################################################
def gobackfrom3D(y):
    y = torch.permute(y, (3, 4, 1, 0, 2))
    y = torch.reshape(y, (y.size(dim=0), y.size(dim=1), y.size(dim=2), -1))
    y = torch.permute(y, (0, 1, 3, 2))
    return y

#######################################################################
# # # Load MAT file
import sys           #Argument
import time
import math as mm
#
path_mat_in = sys.argv[1]
path_mat_out = sys.argv[2]
x = path_mat_in
print('------------------------------------------')
print('Loading file {} ...'.format(x))
t0 = time.time()
x5dtoload = path_mat_in
size = os.stat(x5dtoload).st_size  # in bytes
print('Loading file {} ({:.2f}GB)...'.format(x5dtoload,size/(1024**3)))
zIn = np.load(x5dtoload)
print('x5d loaded')
# input
zIn = torch.tensor(zIn)
print(zIn.type())
zIn = zIn.abs()
print(zIn.shape)
print(zIn.type())
#######################################################################
zIno = zIn
#
#######################################################################
print('=' * 20)
print('-----------ORIGINAL DATA------------------')
#
print(zIn.shape)
print('zIn = torch.permute(zIn, (2,0,1,3,4))')
zIn = torch.permute(zIn, (2,0,1,3,4))
#
print(zIn.shape)
NM = 256
NMx = int((zIn.size(dim=0) - NM)/2)
print(NMx)
#
zIn = zIn[(NMx-1):(NMx-1)+NM,(NMx-1):(NMx-1)+NM,(NMx-1):(NMx-1)+NM,:,:]
print(zIn.shape)
zIn = convertInputMAT256(zIn)
print(zIn.shape)
#
input_test_t = zIn
print('---Variables conv2D---')
print(input_test_t.shape)
#######################################################################
input_test_t = variableforconv3D(input_test_t,NZvolume)
print('---Variables conv3D---')
print(input_test_t.shape)
#
print('** Testing: **')
print(input_test_t.shape)
#
tN = time.time()
print('Total time: {:.6f}sec'.format(tN-t0))
#
test_data = input_test_t
#
del input_test_t

#######################################################################
import torch.nn as nn

#######################################################################
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
class ResidualBlockDown(torch.nn.Module):
    """ Helper Class"""
    def __init__(self, channels):
        super(ResidualBlockDown, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=channels[0],
                            out_channels=channels[1],
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.BatchNorm3d(channels[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(in_channels=channels[1],
                            out_channels=channels[2],
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.BatchNorm3d(channels[2])
        )
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=channels[0],
                            out_channels=channels[2],
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.BatchNorm3d(channels[2])
        )
    def forward(self, x):
        shortcut = x
        block = self.block(x)
        shortcut = self.shortcut(x)
        x = torch.nn.functional.relu(block + shortcut)
        return x
    
class ResidualBlockUp(torch.nn.Module):
    """ Helper Class"""
    def __init__(self, channels):
        super(ResidualBlockUp, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=channels[0],
                                     out_channels=channels[1],
                                     kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1),
                                     padding=1),
            torch.nn.BatchNorm3d(channels[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(in_channels=channels[1],
                            out_channels=channels[2],
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=1),
            torch.nn.BatchNorm3d(channels[2])
        )
        self.shortcut = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=channels[0],
                                     out_channels=channels[2],
                                     kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1),
                                     padding=1),
            torch.nn.BatchNorm3d(channels[2])
        )
    def forward(self, x):
        shortcut = x
        block = self.block(x)
        shortcut = self.shortcut(x)
        x = torch.nn.functional.relu(block + shortcut)
        return x

#######################################################################
Ninputchannels = 64 # 16 coils x 4 motion states
#
import torch.nn.functional as F
# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # # Enconder
        self.residual_block_down_1 = ResidualBlockDown(channels=[Ninputchannels, 128, 128]) # 256x256x128
        self.bn1 = nn.BatchNorm3d(128)
        # self.maxpool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)) # 128x128x128
        self.maxpool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)) # 128x128x128
        self.residual_block_down_2 = ResidualBlockDown(channels=[128, 256, 256]) # 128x128x256
        self.bn2 = nn.BatchNorm3d(256)
        self.maxpool2 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)) # 64x64x256
        self.residual_block_down_3 = ResidualBlockDown(channels=[256, 512, 512]) # 64x64x512
        self.bn3 = nn.BatchNorm3d(512)
        self.maxpool3 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)) # 32x32x512
        self.residual_block_down_4 = ResidualBlockDown(channels=[512, 1024, 1024]) # 32x32x1024
        self.bn4 = nn.BatchNorm3d(1024)
        self.maxpool4 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)) # 16x16x1024
        self.residual_block_down_5 = ResidualBlockDown(channels=[1024, 2048, 2048]) # 16x16x2048
        self.bn5 = nn.BatchNorm3d(2048) # 16x16x2048
        self.up5 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True) # 32x32x2048
        # #
        # # Decoder
        self.residual_block_up_6 = ResidualBlockUp(channels=[1024+2048, 1024, 1024]) # 32x32x1024
        # self.residual_block_up_6 = ResidualBlockUp(channels=[1024, 1024, 1024]) # 32x32x1024
        self.bn6 = nn.BatchNorm3d(1024)
        self.up6 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True) # 64x64x512
        self.residual_block_up_7 = ResidualBlockUp(channels=[512+1024, 512, 512]) # 64x64x512
        # self.residual_block_up_7 = ResidualBlockUp(channels=[512, 512, 512]) # 64x64x512
        self.bn7 = nn.BatchNorm3d(512)
        self.up7 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True) # 128x128x512
        self.residual_block_up_8 = ResidualBlockUp(channels=[256+512, 256, 256]) # 128x128x256
        self.bn8 = nn.BatchNorm3d(256)
        # self.up8 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True) # 256x256x256
        self.up8 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True) # 256x256x256
        self.residual_block_up_9 = ResidualBlockUp(channels=[128+256, 128, 128]) # 256x256x128
        self.bn9 = nn.BatchNorm3d(128)
        #
        # output
        self.convOutput = nn.Conv3d(128, 4, 1, padding=0) # 10 motion states
        #
    def forward(self, x):
        #
        xn = x
        #
        # Encoder
        c1 = self.bn1(self.residual_block_down_1(xn)) # 256x256x128
        p1 = self.maxpool1(c1) # 128x128x128
        c2 = self.bn2(self.residual_block_down_2(p1)) # 128x128x256
        p2 = self.maxpool2(c2) # 64x64x256
        c3 = self.bn3(self.residual_block_down_3(p2)) # 64x64x512
        p3 = self.maxpool3(c3) # 32x32x512
        c4 = self.bn4(self.residual_block_down_4(p3)) # 32x32x1024
        p4 = self.maxpool4(c4) # 16x16x1024
        c5 = self.bn5(self.residual_block_down_5(p4)) # 16x16x2048
        # #
        # Decoder
        p5 = self.up5(c5) # 32x32x2048
        u6 = torch.cat((p5, c4), 1) #32x32x(2048+1024)
        c6 = self.bn6(self.residual_block_up_6(u6)) # 32x32x1024
        p6 = self.up6(c6) # 64x64x1024
        u7 = torch.cat((p6, c3), 1) # 64x64x(1024+512)
        c7 = self.bn7(self.residual_block_up_7(u7)) # 64x64x512
        p7 = self.up7(c7) # 128x128x512
        u8 = torch.cat((p7, c2), 1) # 128x128x(512+256)
        c8 = self.bn8(self.residual_block_up_8(u8)) # 128x128x256
        p8 = self.up8(c8) # 256x256x256
        u9 = torch.cat((p8, c1), 1) # 256x256x(256+128)
        c9 = self.bn9(self.residual_block_up_9(u9)) # 256x256x128
        #
        # Output
        x = self.convOutput(c9)
        x = F.relu(x)
        # #
        return x
model = Net()
del Net
# # move tensors to GPU if CUDA is available
#######################################################################
if train_on_gpu:
    # model.cuda()
    model= nn.DataParallel(model)
model.to(device)
print(device)
#
#
from collections import OrderedDict
state_dict = torch.load(name2load, map_location='cpu')
state_dict = state_dict['model_state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if train_on_gpu:
        new_state_dict[k] = v
    else:
        new_state_dict[k.replace("module.", "")] = v # this worked with CPU only
model.load_state_dict(new_state_dict)
#
# # Go to evaluation mode
model.eval()
outresults = torch.tensor([])  # For results
with torch.no_grad():
    if train_on_gpu:
        test_data = test_data.cuda()
    tTt0 = time.time()
    outresults = model(test_data)
    tTt = time.time()
#
outresults = gobackfrom3D(outresults)
#
if train_on_gpu:
    outresults = outresults.to('cpu')  # Move results to CPU
#
print('Time Testing ({} slices, no saving in HD): {:.6f}sec'.format(outresults.size(dim=0), tTt-tTt0))
outresults_ori = outresults
del outresults
#######################################################################
outresults_ori = outresults_ori[:,:,0:NM,:]
# # Save results
from scipy.io import savemat
var2save = {"test_pd_ori":outresults_ori.detach().numpy()}
savemat(path_mat_out,var2save, do_compression=False)
