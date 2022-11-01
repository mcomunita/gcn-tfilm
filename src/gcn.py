import torch
import math
import argparse
import src.utils as utils
# from contextlib import nullcontext


def wrapperkwargs(func, kwargs):
    return func(**kwargs)


def wrapperargs(func, args):
    return func(*args)


# def load_model(model_data):
#     model_types = {'GCN': GCN}

#     model_meta = model_data.pop('model_data')

#     if model_meta['model'] == 'SimpleRNN' or model_meta['model'] == 'GatedConvNet':
#         network = wrapperkwargs(
#             model_types[model_meta.pop('model')], model_meta)
#         if 'state_dict' in model_data:
#             state_dict = network.state_dict()
#             for each in model_data['state_dict']:
#                 state_dict[each] = torch.tensor(model_data['state_dict'][each])
#             network.load_state_dict(state_dict)

#     elif model_meta['model'] == 'RecNet':
#         model_meta['blocks'] = []
#         network = wrapperkwargs(
#             model_types[model_meta.pop('model')], model_meta)
#         for i in range(len(model_data['blocks'])):
#             network.add_layer(model_data['blocks'][str(i)])

#         # Get the state dict from the newly created model and load the saved states, if states were saved
#         if 'state_dict' in model_data:
#             state_dict = network.state_dict()
#             for each in model_data['state_dict']:
#                 state_dict[each] = torch.tensor(model_data['state_dict'][each])
#             network.load_state_dict(state_dict)

#         if 'training_info' in model_data.keys():
#             network.training_info = model_data['training_info']

#     return network


# This is a function for taking the old json config file format I used to use and converting it to the new format
# def legacy_load(legacy_data):
#     if legacy_data['unit_type'] == 'GRU' or legacy_data['unit_type'] == 'LSTM':
#         model_data = {'model_data': {
#             'model': 'RecNet', 'skip': 0}, 'blocks': {}}
#         model_data['blocks']['0'] = {
#             'block_type': legacy_data['unit_type'],
#             'input_size': legacy_data['in_size'],
#             'hidden_size': legacy_data['hidden_size'],
#             'output_size': 1,
#             'lin_bias': True
#         }
#         if legacy_data['cur_epoch']:
#             training_info = {
#                 'current_epoch': legacy_data['cur_epoch'],
#                 'training_losses': legacy_data['tloss_list'],
#                 'val_losses': legacy_data['vloss_list'],
#                 'load_config': legacy_data['load_config'],
#                 'low_pass': legacy_data['low_pass'],
#                 'val_freq': legacy_data['val_freq'],
#                 'device': legacy_data['pedal'],
#                 'seg_length': legacy_data['seg_len'],
#                 'learning_rate': legacy_data['learn_rate'],
#                 'batch_size': legacy_data['batch_size'],
#                 'loss_func': legacy_data['loss_fcn'],
#                 'update_freq': legacy_data['up_fr'],
#                 'init_length': legacy_data['init_len'],
#                 'pre_filter': legacy_data['pre_filt']
#             }
#             model_data['training_info'] = training_info

#         if 'state_dict' in legacy_data:
#             state_dict = legacy_data['state_dict']
#             state_dict = dict(state_dict)
#             new_state_dict = {}
#             for each in state_dict:
#                 new_name = each[0:7] + 'block_1.' + each[9:]
#                 new_state_dict[new_name] = state_dict[each]
#             model_data['state_dict'] = new_state_dict
#         return model_data
#     else:
#         print('format not recognised')


""" 
Gated convolutional layer, zero pads and then applies a causal convolution to the input 
"""


class GatedConv1d(torch.nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size

        # Layers: Conv1D -> Activations -> Mix + Residual

        self.conv = torch.nn.Conv1d(in_channels=in_ch,
                                    out_channels=out_ch * 2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=dilation)

        self.mix = torch.nn.Conv1d(in_channels=out_ch,
                                   out_channels=out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x):
        # print("GatedConv1d: ", x.shape)
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, :self.out_ch, :]) * \
            torch.sigmoid(y[:, self.out_ch:, :])

        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0],
                                   self.out_ch,
                                   residual.shape[2] - z.shape[2]),
                       z),
                      dim=2)

        x = self.mix(z) + residual

        return x, z


""" 
Gated convolutional neural net block, applies successive gated convolutional layers to the input, a total of 'layers'
layers are applied, with the filter size 'kernel_size' and the dilation increasing by a factor of 'dilation_growth' for
each successive layer.
"""


class GCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 nlayers,
                 kernel_size,
                 dilation_growth):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size))
            in_ch = out_ch

    def forward(self, x):
        # print("GCNBlock: ", x.shape)
        # [batch, channels, length]
        z = torch.empty([x.shape[0],
                         self.nlayers * self.out_ch,
                         x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x)
            z[:, n * self.out_ch: (n + 1) * self.out_ch, :] = zn

        return x, z


""" 
Gated Convolutional Neural Net class, based on the 'WaveNet' architecture, takes a single channel of audio as input and
produces a single channel of audio of equal length as output. one-sided zero-padding is used to ensure the network is 
causal and doesn't reduce the length of the audio.

Made up of 'blocks', each one applying a series of dilated convolutions, with the dilation of each successive layer 
increasing by a factor of 'dilation_growth'. 'layers' determines how many convolutional layers are in each block,
'kernel_size' is the size of the filters. Channels is the number of convolutional channels.

The output of the model is creating by the linear mixer, which sums weighted outputs from each of the layers in the 
model
"""


class GCN(torch.nn.Module):
    def __init__(self,
                 nblocks=2,
                 nlayers=9,
                 nchannels=8,
                 kernel_size=3,
                 dilation_growth=2,
                 **kwargs):
        super(GCN, self).__init__()
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch=1 if b == 0 else nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x):
        # print("GCN: ", x.shape)
        # x.shape = [length, batch, channels]
        x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])

        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)
            z[:,
              n * self.nchannels * self.nlayers:
              (n + 1) * self.nchannels * self.nlayers,
              :] = zn

        # back to [length, batch, channels]
        return self.blocks[-1](z).permute(2, 0, 1)

    # train_epoch runs one epoch of training
    def train_epoch(self,
                    input_data,
                    target_data,
                    loss_fcn,
                    optim,
                    batch_size):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # iterate over the batches
        ep_losses = None

        for batch_i in range(math.ceil(shuffle.shape[0] / batch_size)):
            # zero all gradients
            self.zero_grad()

            # load batch
            input_batch = input_data[:,
                                     shuffle[batch_i * batch_size:
                                             (batch_i + 1) * batch_size],
                                     :]
            target_batch = target_data[:,
                                       shuffle[batch_i * batch_size:
                                               (batch_i + 1) * batch_size],
                                       :]

            # process batch
            output = self(input_batch)

            # loss and backprop
            batch_losses = loss_fcn(output, target_batch)

            tot_batch_loss = 0
            for loss in batch_losses:
                tot_batch_loss += batch_losses[loss]

            tot_batch_loss.backward()
            optim.step()

            # add batch losses to epoch losses
            for loss in batch_losses:
                if ep_losses == None:
                    ep_losses = batch_losses
                else:
                    ep_losses[loss] += batch_losses[loss]

        # mean epoch losses
        for loss in ep_losses:
            ep_losses[loss] /= (batch_i + 1)

        return ep_losses

    def process_data(self,
                     input_data,
                     target_data=None,
                     chunk=16384,
                     loss_fcn=None,
                     grad=False):

        rf = self.compute_receptive_field()

        if not (input_data.shape[0] / chunk).is_integer():
            # round to next chunk size
            padding = chunk - (input_data.shape[0] % chunk)
            input_data = torch.nn.functional.pad(input_data,
                                                 (0, 0, 0, 0, 0, padding),
                                                 mode='constant',
                                                 value=0)
            if target_data != None:
                target_data = torch.nn.functional.pad(target_data,
                                                      (0, 0, 0, 0, 0, padding),
                                                      mode='constant',
                                                      value=0)

        with torch.no_grad():
            # process input
            output_data = torch.empty_like(input_data)

            for l in range(int(output_data.size()[0] / chunk)):
                input_chunk = input_data[l * chunk: (l + 1) * chunk]
                if l == 0:  # first chunk
                    padding = torch.zeros([rf, input_chunk.shape[1], input_chunk.shape[2]])
                else:
                    padding = input_data[(l * chunk) - rf: l * chunk]
                input_chunk = torch.cat([padding, input_chunk])
                output_chunk = self(input_chunk)
                output_data[l * chunk: (l + 1) * chunk] = \
                    output_chunk[rf:, :, :]

            if loss_fcn != None and target_data != None:
                losses = loss_fcn(output_data, target_data)
                return input_data, target_data, output_data, losses

        return input_data, target_data, output_data

    def save_model(self,
                   file_name,
                   direc=""):
        if direc:
            utils.dir_check(direc)

        model_data = {"model_data": {"model_type": "gcn",
                                     "nblocks": self.nblocks,
                                     "nlayers": self.nlayers,
                                     "nchannels": self.nchannels,
                                     "kernel_size": self.kernel_size,
                                     "dilation_growth": self.dilation_growth}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data["state_dict"] = model_state

        utils.json_save(model_data, file_name, direc)

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--nblocks', type=int, default=2)
        parser.add_argument('--nlayers', type=int, default=9)
        parser.add_argument('--nchannels', type=int, default=16)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--dilation_growth', type=int, default=2)

        return parser
