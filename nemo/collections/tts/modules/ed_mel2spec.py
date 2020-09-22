# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types.elements import IntType, LengthsType, SpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental

from numpy.linalg import pinv

def str2act(txt):
    """Translates text to neural network activation"""
    return {
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "none": nn.Sequential(),
        "lrelu": nn.LeakyReLU(0.1),
        "selu": nn.SELU(),
    }[txt.lower()]

class OperationMode(Enum):
    """Training or Inference (Evaluation) mode"""

    training = 0
    validation = 1
    infer = 2


def create_mel_filterbank(*args, **kwargs):
    return librosa.filters.mel(*args, **kwargs)

@experimental
class EDMel2SpecModule(NeuralModule, Exportable):
    def __init__(self, n_fft: int, hop_length: int, mel_fmin:int, mel_fmax:int, mel_freq:int, layers:int, sampling_rate:int , subseq_len:int, ngf:int, use_batchnorm:bool, 
                droprate:float,  num_dropout:int, pre_final_lin: bool, act1: str, act2:str, use_weight_norm:bool):
        """
        Degli module

        Args:
            n_fft (int): STFT argument.
            hop_length (int): STFT argument.
            depth (int): depth > 1 will cause the NN to repeat, which new trainable weights.
            out_all_block (bool) output all blocks, not just the final output.
            tiny (bool) override all model hyperparameters and use the paper's small model.
            kwargs: to be passed for the dnn.
        """


        super().__init__()

        n_freq = n_fft // 2 + 1
        self.hop_length = hop_length

        self.n_layers = layers
        self.sampling_rate = sampling_rate
        self.subseq_len = subseq_len
        self.ngf = ngf
        self.use_batchnorm = use_batchnorm
        self.droprate = droprate
        self.num_dropout = num_dropout
        self.pre_final_lin = pre_final_lin
        self.act1 = act1
        self.act2 = act2
        self.use_weight_norm = use_weight_norm

        meltrans = create_mel_filterbank(sampling_rate, n_fft, mel_fmin, mel_fmax, mel_freq)

        self.meltrans = nn.Parameter(torch.transpose(torch.tensor(meltrans, dtype=torch.float),0,1), requires_grad = False) 
        self.meltrans_inv = nn.Parameter(torch.transpose(torch.tensor(pinv(meltrans), dtype=torch.float),0,1), requires_grad = False) 

        layer_specs = [
            1, 
            self.ngf, 
            self.ngf * 2, 
            self.ngf * 4, 
            self.ngf * 8, 
            self.ngf * 8,
            self.ngf * 8, 
            self.ngf * 8,
            self.ngf * 8,
            self.ngf * 8,
        ]

        layer_specs = layer_specs[0:self.n_layers+1]
        self.encoders = nn.ModuleList()

        conv, pad = self._gen_conv(layer_specs[0] ,layer_specs[1], use_weight_norm = self.use_weight_norm)
        self.encoders.append(nn.Sequential(pad, conv))
        
        last_ch = layer_specs[1]

        for i,ch_out in enumerate(layer_specs[2:]):
            d = OrderedDict()
            d['act'] = str2act(self.act1)
            gain  = math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2)

            conv, pad  = self._gen_conv(last_ch ,ch_out, gain = gain, use_weight_norm = self.use_weight_norm)

            d['pad'] = pad
            d['conv'] = conv

            if self.use_batchnorm:
                d['bn']  = nn.BatchNorm2d(ch_out)

            encoder_block = nn.Sequential(d)
            self.encoders.append(encoder_block)
            last_ch = ch_out

        layer_specs.reverse()
        self.decoders = nn.ModuleList()
        for i,ch_out in enumerate(layer_specs[1:]):

            d = OrderedDict()
            d['act'] = str2act(self.act2)
            gain  =  math.sqrt(2.0/(1.0+self.lamb**2))
            gain = gain / math.sqrt(2) 
            
            kernel_size = 4 if i < len(layer_specs)-2 else 5
            conv = self._gen_deconv(last_ch, ch_out , gain = gain, k= kernel_size, use_weight_norm = self.use_weight_norm)
            d['conv'] = conv

            if i < self.num_dropout and self.droprate > 0.0:
                d['dropout'] = nn.Dropout(self.droprate)

            if self.use_batchnorm and i < self.n_layers-1:
                d['bn']  = nn.BatchNorm2d(ch_out)

            decoder_block = nn.Sequential(d)
            self.decoders.append(decoder_block)
            last_ch = ch_out * 2

        init_alpha = 0.001
        self.linear_finalizer = nn.Parameter(torch.ones(n_freq) * init_alpha , requires_grad = True)
    
        if self.pre_final_lin:
            self.linear_pre_final = nn.Parameter(torch.ones(self.ngf*2, n_freq//2) , requires_grad = True)

    def mel_pseudo_inverse(self,x):
        return torch.tensordot(x,self.meltrans_inv, dims=[[2],[0]]).permute(0,1,3,2)

    def spec_to_mel(self,x):
        return torch.tensordot(x,self.meltrans, dims=[[2],[0]]).permute(0,1,3,2)


    @typecheck()
    def forward(self, mel):
        x = self.mel_pseudo_inverse(mel)
        x_in = x

        encoders_output = []

        for i,encoder in enumerate(self.encoders):
            x = encoder(x)
            encoders_output.append(x)

        for i,decoder in enumerate(self.decoders[:-1]):
            x = decoder(x)            
            x = torch.cat([x, encoders_output[-(i+2)]], dim=1)

        if self.pre_final_lin:
            x_perm = x.permute(0,3,1,2)
            x = torch.mul(x_perm,  self.linear_pre_final).permute(0,2,3,1)

        x = self.decoders[-1](x) 
        x_perm = x.permute(0,1,3,2)
        x = torch.mul(x_perm,  self.linear_finalizer) 
        x = x.permute(0,1,3,2)

        x = x + x_in 
        return x

    def _gen_conv(self, in_ch,  out_ch, strides = (2, 1), kernel_size = (5,3), gain = math.sqrt(2), pad = (1,1,1,2) , use_weight_norm = False  ):
        pad = torch.nn.ReplicationPad2d(pad)
        conv =  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride = strides , padding=0)

        if use_weight_norm:
            conv =  torch.nn.utils.weight_norm( conv  ,name='weight')
            
        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv, pad 

    def _gen_deconv(self, in_ch,  out_ch, strides = (2, 1), k = 4, gain = math.sqrt(2), p =1 , use_weight_norm = False ):
        conv =  nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(k,3), stride = strides, padding_mode='zeros',padding = (p,1), dilation  = 1)

        if use_weight_norm:
            conv =  torch.nn.utils.weight_norm( conv  ,name='weight')

        w = conv.weight
        k = w.size(1) * w.size(2) * w.size(3)
        conv.weight.data.normal_(0.0, gain / math.sqrt(k) )
        nn.init.constant_(conv.bias,0.01)
        return conv


    @property
    def input_types(self):
        return {
            "mel": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "spec": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    def input_example(self):
        # TODO: Implement me!
        pass

    def save_to(self, save_path: str):
        # TODO: Implement me!
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        # TODO: Implement me!
        pass
