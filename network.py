import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from distutils import util
import numpy as np

'''
def build_emulator(dim_in=25, num_neurons=300, num_pixel=7214, emulator_coeffs=None, use_cuda=True):
    # Create layers
    model = torch.nn.Sequential(torch.nn.Linear(dim_in, num_neurons),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(num_neurons, num_neurons),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(num_neurons, num_pixel))
    # Assign pre-trained weights
    if emulator_coeffs is not None:
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, y_min, y_max = emulator_coeffs
        y_min = torch.Tensor(y_min.astype(np.float32))
        y_max = torch.Tensor(y_max.astype(np.float32))
        
        for param, weights in zip(model.parameters(), 
                                  [w_array_0, b_array_0, w_array_1, b_array_1, w_array_2, b_array_2]):
            param.data =  Variable(torch.from_numpy(weights)).type(torch.FloatTensor)
            
        if use_cuda:
            model = model.cuda()
            y_min = y_min.cuda()
            y_max = y_max.cuda()
            
        return model, y_min, y_max
    
    else:
        if use_cuda:
            model = model.cuda()
        return model
'''    
def build_emulator(model_fn, dim_in=25, num_neurons=300, num_pixel=7214, use_cuda=True):
    
    # Create layers
    model = torch.nn.Sequential(torch.nn.Linear(dim_in, num_neurons),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(num_neurons, num_neurons),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(num_neurons, num_pixel))
    
    # Load model info
    checkpoint = torch.load(model_fn, map_location=lambda storage, loc: storage)
    y_min = checkpoint['y_min']
    y_max = checkpoint['y_max']
    
    # Load model weights
    model.load_state_dict(checkpoint['Payne'])
    
    # Change to GPU
    if use_cuda:
        model = model.cuda()
        y_min = y_min.cuda()
        y_max = y_max.cuda()
    
    return model, y_min, y_max
    
def init_weights(m):
    """
    Glorot uniform initialization for network.
    """
    if 'conv' in m.__class__.__name__.lower():
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """
    
    f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
    return f.size()[1:]

def same_padding(input_pixels, filter_len, stride=1):
    effective_filter_size_rows = (filter_len - 1) + 1
    output_pixels = (input_pixels + stride - 1) // stride
    padding_needed = max(0, (output_pixels - 1) * stride + 
                         effective_filter_size_rows - input_pixels)
    padding = max(0, (output_pixels - 1) * stride +
                        (filter_len - 1) + 1 - input_pixels)
    rows_odd = (padding % 2 != 0)    
    return padding // 2
    
class Encoder(nn.Module):

    def __init__(self, num_pixels, num_y, num_z, conv_filts, 
                 conv_strides, conv_filt_lens, activ_fn,
                 init=True, use_cuda=True):
                
        super(Encoder, self).__init__()
        
        # Build conv layers
        layers = []
        input_filts = 1
        for filts, strides, filter_length in zip(conv_filts, conv_strides, conv_filt_lens):   
            layers.append(nn.Conv1d(input_filts, filts, filter_length, strides))
            layers.append(activ_fn)
            input_filts = filts
        self.conv_layers = torch.nn.Sequential(*layers)

        # Calculate the size of the conv output
        self.conv_output_shape = compute_out_size((1, num_pixels), self.conv_layers)
        fc_input_dim = np.prod(list(self.conv_output_shape))

        # Output fully-connected layers
        self.fc_y = nn.Linear(fc_input_dim, num_y)
        self.fc_z = nn.Linear(fc_input_dim, num_z)

        # Initialize weights and biases
        if init:
            self.conv_layers.apply(init_weights)
            self.fc_y.apply(init_weights)
            self.fc_z.apply(init_weights)

        # Change model to GPU
        if use_cuda:
            self.conv_layers = self.conv_layers.cuda()
            self.fc_y = self.fc_y.cuda()
            self.fc_z = self.fc_z.cuda()

    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc_y(conv_out), self.fc_z(conv_out)

class Decoder(nn.Module):

    def __init__(self, conv_input_shape, num_pixels, num_y, num_z, conv_filts, 
                 conv_strides, conv_filt_lens, activ_fn,
                 init=True, use_cuda=True):
                
        super(Decoder, self).__init__()
        self.conv_input_shape = conv_input_shape
        
        # Fully-connected input
        fc_input_dim = num_y+num_z
        fc_output_dim = np.prod(list(self.conv_input_shape))
        self.fc_in = nn.Linear(fc_input_dim, fc_output_dim)
        
        # Build conv layers (reverse order of encoders)
        layers = []
        input_filts = self.conv_input_shape[0]
        for filts, strides, filter_length in zip(reversed(conv_filts), reversed(conv_strides), reversed(conv_filt_lens)):
            if strides>1:
                layers.append(nn.ConvTranspose1d(input_filts, filts, filter_length, strides))
            else:      
                layers.append(nn.Conv1d(input_filts, filts, filter_length, strides))
            layers.append(activ_fn)
            input_filts = filts
        # Spectrum output
        layers.append(nn.Conv1d(input_filts, 1, 1, 1))
        self.conv_layers = torch.nn.Sequential(*layers)

        if init:
            # Initialize weights and biases
            self.conv_layers.apply(init_weights)
            self.fc_in.apply(init_weights)

        if use_cuda:
            self.conv_layers = self.conv_layers.cuda()
            self.fc_in = self.fc_in.cuda()
        
    def forward(self, yz):
        fc_out = self.fc_in(yz).view((yz.size()[0],*self.conv_input_shape))
        return self.conv_layers(fc_out)

class SN_AE(nn.Module):

    def __init__(self, architecture_config, emulator_fn='model/PAYNE.pth.tar', use_cuda=True):
                
        super(SN_AE, self).__init__()
                
        # Read configuration
        num_pixels = int(architecture_config['num_pixels'])
        activation = architecture_config['activation']
        conv_filts = eval(architecture_config['conv_filts'])
        conv_filt_lens = eval(architecture_config['conv_filt_lens'])
        conv_strides = eval(architecture_config['conv_strides'])
        self.num_y = int(architecture_config['num_y'])
        self.num_z = int(architecture_config['num_z'])
        
        # Create emulator
        (self.emulator, self.y_min, self.y_max) = build_emulator(model_fn=emulator_fn, 
                                                                 use_cuda=use_cuda)
        
        # Define activation function
        if activation.lower()=='sigmoid':
            activ_fn = torch.nn.Sigmoid()
        elif activation.lower()=='leakyrelu':
            activ_fn = torch.nn.LeakyReLU(0.1)
        elif activation.lower()=='relu':
            activ_fn = torch.nn.ReLU()
        
        # Build encoding network
        self.encoder = Encoder(num_pixels, 
                               self.num_y, self.num_z,
                               conv_filts, conv_strides, 
                               conv_filt_lens, activ_fn, 
                               init=True, use_cuda=use_cuda)
        
        # Build decoding network    
        self.decoder = Decoder(self.encoder.conv_output_shape,
                               num_pixels, 
                               self.num_y, self.num_z,
                               conv_filts, conv_strides, 
                               conv_filt_lens, activ_fn, 
                               init=True, use_cuda=use_cuda)
        
        if use_cuda:
            self.y_min = self.y_min.cuda()
            self.y_max = self.y_max.cuda()
        
    def run_emulator(self, y):
        # Normalize labels
        y = (y - self.y_min)/(self.y_max-self.y_min) - 0.5
        return self.emulator(y)
    
    def x_to_yz(self, x):        
        return self.encoder(x.unsqueeze(1))
    
    def yz_to_xsynth(self, y, raw_labels=True):
        # Use zeros as z
        z = torch.ones((y.size()[0], self.num_z), dtype=torch.float32)
        if raw_labels:
            # Normalize labels
            y = (y - self.y_min)/(self.y_max-self.y_min) - 0.5
        return self.decoder(torch.cat((y, z), 1)).squeeze(1)
    
    def yz_to_xobs(self, y, z, raw_labels=False):
        if raw_labels:
            # Normalize labels
            y = (y - self.y_min)/(self.y_max-self.y_min) - 0.5
        return self.decoder(torch.cat((y, z), 1)).squeeze(1)
        
    def train_mode(self):
        self.emulator.eval()
        self.encoder.train()
        self.decoder.train()
        
    def eval_mode(self):
        self.emulator.eval()
        self.encoder.eval()
        self.decoder.eval()
    
    def forward(self, x, synth=True):
        y, z = self.x_to_yz(x)
        if synth:
            return self.yz_to_xsynth(y)
        else:
            return self.yz_to_xobs(y, z)
