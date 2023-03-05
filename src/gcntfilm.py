import torch
import math
import argparse
import src.utils as utils


"""
Temporal FiLM layer - Conditional
"""
class TFiLM(torch.nn.Module):
    def __init__(self,
                 nchannels,
                 nparams,
                 block_size):
        super(TFiLM, self).__init__()
        self.nchannels = nchannels
        self.nparams = nparams
        self.block_size = block_size
        self.num_layers = 1
        self.hidden_state = None  # (hidden_state, cell_state)

        # used to downsample input
        self.maxpool = torch.nn.MaxPool1d(kernel_size=block_size,
                                          stride=None,
                                          padding=0,
                                          dilation=1,
                                          return_indices=False,
                                          ceil_mode=False)

        self.lstm = torch.nn.LSTM(input_size=nchannels+nparams,
                                  hidden_size=nchannels,
                                  num_layers=self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

    def forward(self, x, p=None):
        # x = [batch, nchannels, length]
        # p = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding = torch.zeros(x_in_shape[0], x_in_shape[1], self.block_size - (x_in_shape[2] % self.block_size))
            x = torch.cat((x, padding), dim=-1)

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.maxpool(x)

        if self.nparams > 0 and p != None:
            p_up = p.unsqueeze(-1).repeat(1, 1, nsteps) # upsample params [batch, nparams, nsteps]
            x_down = torch.cat((x_down, p_up), dim=1) # concat along channel dim [batch, nchannels+nparams, nsteps]

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state == None:  # state was reset
            # init hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.nchannels).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.nchannels).requires_grad_()
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, nsteps, self.block_size))
        x_norm = torch.reshape(
            x_norm, shape=(-1, self.nchannels, nsteps, 1))

        # multiply
        x_out = x_norm * x_in

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., :x_in_shape[2]]

        return x_out

    # def detach_state(self):
    #     if self.hidden_state.__class__ == tuple:
    #         self.hidden_state = tuple([h.clone().detach() for h in self.hidden_state])
    #     else:
    #         self.hidden_state = self.hidden_state.clone().detach()

    def reset_state(self):
        # print("Reset Hidden State")
        self.hidden_state = None


""" 
Gated convolutional layer, zero pads and then applies a causal convolution to the input
"""
class GatedConv1d(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size,
                 nparams,
                 tfilm_block_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        # Layers: Conv1D -> Activations -> TFiLM -> Mix + Residual

        self.conv = torch.nn.Conv1d(in_channels=in_ch,
                                    out_channels=out_ch * 2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=dilation)

        self.tfilm = TFiLM(nchannels=out_ch,
                           nparams=nparams,
                           block_size=tfilm_block_size)

        self.mix = torch.nn.Conv1d(in_channels=out_ch,
                                   out_channels=out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x, p=None):
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

        # modulation
        z = self.tfilm(z, p)

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
                 dilation_growth,
                 nparams,
                 tfilm_block_size):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size,
                                           nparams=nparams,
                                           tfilm_block_size=tfilm_block_size))
            in_ch = out_ch

    def forward(self, x, p=None):
        # print("GCNBlock: ", x.shape)
        # [batch, channels, length]
        z = torch.empty([x.shape[0],
                         self.nlayers * self.out_ch,
                         x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x, p)
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
class GCNTF(torch.nn.Module):
    def __init__(self,
                 nparams=0,
                 nblocks=2,
                 nlayers=9,
                 nchannels=8,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=128,
                 device="cpu",
                 **kwargs):
        super(GCNTF, self).__init__()
        self.nparams = nparams
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size
        self.device = device

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch=1 if b == 0 else nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        nparams=nparams,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x, p=None):
        # print("GCN: ", x.shape)
        # x.shape = [length, batch, channels]
        # x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])

        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x, p)
            z[:,
              n * self.nchannels * self.nlayers:
              (n + 1) * self.nchannels * self.nlayers,
              :] = zn

        # back to [length, batch, channels]
        # return self.blocks[-1](z).permute(2, 0, 1)
        return self.blocks[-1](z)

    # def detach_states(self):
    #     # print("DETACH STATES")
    #     for layer in self.modules():
    #         if isinstance(layer, TFiLM):
    #             layer.detach_state()

    # reset state for all TFiLM layers
    def reset_states(self):
        # print("RESET STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()


    # train_epoch runs one epoch of training
    def train_epoch(self,
                    dataloader,
                    loss_fcn,
                    optimiser):
        # print("TRAIN EPOCH")
        ep_losses = None

        for batch_idx, batch in enumerate(dataloader):
            # print("TRAIN BATCH")
            # reset states before starting new batch
            self.reset_states()

            # zero all gradients
            self.zero_grad()

            input, target, params = batch
            input = input.to(self.device)
            target = target.to(self.device)
            params = params.to(self.device)

            # process batch
            pred = self(input, params)
            pred = pred.to(self.device)

            # loss and backprop
            batch_losses = loss_fcn(pred, target)

            tot_batch_loss = 0
            for loss in batch_losses:
                tot_batch_loss += batch_losses[loss]

            tot_batch_loss.backward()
            optimiser.step()

            # add batch losses to epoch losses
            for loss in batch_losses:
                if ep_losses == None:
                    ep_losses = batch_losses
                else:
                    ep_losses[loss] += batch_losses[loss]

        # mean epoch losses
        for loss in ep_losses:
            ep_losses[loss] /= (batch_idx + 1)

        return ep_losses
    

    def val_epoch(self,
                  dataloader,
                  loss_fcn):
        val_losses = None

        # evaluation mode
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # reset states before starting new batch
                self.reset_states()

                input, target, params = batch
                input = input.to(self.device)
                target = target.to(self.device)
                params = params.to(self.device)

                # process batch
                pred = self(input, params)
                pred = pred.to(self.device)

                # loss
                batch_losses = loss_fcn(pred, target)

                tot_batch_loss = 0
                for loss in batch_losses:
                    tot_batch_loss += batch_losses[loss]

                # add batch losses to epoch losses
                for loss in batch_losses:
                    if val_losses == None:
                        val_losses = batch_losses
                    else:
                        val_losses[loss] += batch_losses[loss]

        # mean val losses
        for loss in val_losses:
            val_losses[loss] /= (batch_idx + 1)
        
        # back to training mode
        self.train()
        return val_losses


    def test_epoch(self,
                  dataloader,
                  loss_fcn):
        test_losses = None

        # evaluation mode
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # reset states before starting new batch
                self.reset_states()

                input, target, params = batch
                input = input.to(self.device)
                target = target.to(self.device)
                params = params.to(self.device)

                # process batch
                pred = self(input, params)
                pred = pred.to(self.device)

                # loss
                batch_losses = loss_fcn(pred, target)

                tot_batch_loss = 0
                for loss in batch_losses:
                    tot_batch_loss += batch_losses[loss]

                # add batch losses to epoch losses
                for loss in batch_losses:
                    if test_losses == None:
                        test_losses = batch_losses
                    else:
                        test_losses[loss] += batch_losses[loss]

        # mean val losses
        for loss in test_losses:
            test_losses[loss] /= (batch_idx + 1)
        
        # back to training mode
        self.train()
        return test_losses


    def process_data(self,
                     input,
                     params):
        
        input = input.to(self.device)
        params = params.to(self.device)
        
        # evaluation mode
        self.eval()
        with torch.no_grad():
            # reset states before processing
            self.reset_states()

            out = self(x=input, p=params)
        
        # back to training mode
        self.train()

        # reset states before other computations
        self.reset_states()

        return out


    def save_model(self,
                   file_name,
                   direc=""):
        if direc:
            utils.dir_check(direc)

        model_data = {"model_data": {"model_type": "gcntf",
                                     "nparams": self.nparams,
                                     "nblocks": self.nblocks,
                                     "nlayers": self.nlayers,
                                     "nchannels": self.nchannels,
                                     "kernel_size": self.kernel_size,
                                     "dilation_growth": self.dilation_growth,
                                     "tfilm_block_size": self.tfilm_block_size}}

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
        parser.add_argument('--nparams', type=int, default=0)
        parser.add_argument('--nblocks', type=int, default=2)
        parser.add_argument('--nlayers', type=int, default=9)
        parser.add_argument('--nchannels', type=int, default=16)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--dilation_growth', type=int, default=2)
        parser.add_argument('--tfilm_block_size', type=int, default=128)

        return parser
