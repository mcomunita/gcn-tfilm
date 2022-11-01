import torch
import math
import argparse
import src.utils as utils
# from contextlib import nullcontext


def wrapperkwargs(func, kwargs):
    return func(**kwargs)


def wrapperargs(func, args):
    return func(*args)


"""
Simple RNN class made of a single recurrent unit of type LSTM, GRU or Elman, 
followed by a fully connected layer
"""


class RNN(torch.nn.Module):
    def __init__(self,
                 input_size=1,
                 output_size=1,
                 unit_type="LSTM",
                 hidden_size=32,
                 skip=1,
                 bias_fl=True,
                 nlayers=1,
                 **kwargs):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.unit_type = unit_type
        self.hidden_size = hidden_size
        self.skip = skip
        self.bias_fl = bias_fl
        self.nlayers = nlayers
        self.save_state = True
        self.hidden = None  # hidden state

        # create dictionary of possible block types
        self.rec = wrapperargs(getattr(torch.nn, unit_type),
                               [input_size, hidden_size, nlayers])

        self.lin = torch.nn.Linear(hidden_size, output_size, bias=bias_fl)

    def forward(self, x):
        if self.skip:
            # save the residual for the skip connection
            res = x[:, :, :self.skip]
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x) + res
        else:
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # reset hidden state
    def reset_hidden(self):
        self.hidden = None

    # train_epoch runs one epoch of training
    def train_epoch(self,
                    input_data,
                    target_data,
                    loss_fcn,
                    optim,
                    batch_size,
                    init_len=200,
                    up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # iterate over the batches
        ep_losses = {}

        for batch_i in range(math.ceil(shuffle.shape[0] / batch_size)):
            # load batch
            input_batch = input_data[:,
                                     shuffle[batch_i * batch_size:
                                             (batch_i + 1) * batch_size],
                                     :]
            target_batch = target_data[:,
                                       shuffle[batch_i * batch_size:
                                               (batch_i + 1) * batch_size],
                                       :]

            # initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            tot_batch_losses = {}

            # iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # process batch
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # loss and backprop
                partial_batch_losses = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])

                partial_batch_loss = 0
                for loss in partial_batch_losses:
                    partial_batch_loss += partial_batch_losses[loss]

                partial_batch_loss.backward()
                optim.step()

                # detach hidden state for truncated BPTT
                self.detach_hidden()
                self.zero_grad()

                # update the start index for next iteration
                start_i += up_fr

                # add partial batch losses to total batch losses
                if tot_batch_losses == {}:
                    tot_batch_losses = partial_batch_losses
                else:
                    for loss in partial_batch_losses:
                        tot_batch_losses[loss] += partial_batch_losses[loss]

            # add average batch losses to epoch losses
            if ep_losses == {}:
                for loss in tot_batch_losses:
                    ep_losses[loss] = tot_batch_losses[loss] / (k + 1)
            else:
                for loss in tot_batch_losses:
                    ep_losses[loss] += tot_batch_losses[loss] / (k + 1)

            # # add the average batch loss to the epoch loss and reset the hidden states to zeros
            # ep_loss += batch_loss / (k + 1)

            # reset hidden state before next batch
            self.reset_hidden()

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
            # reset state before processing
            self.reset_hidden()

            # process input
            output_data = torch.empty_like(input_data)

            for l in range(int(output_data.size()[0] / chunk)):
                output_data[l * chunk:(l + 1) * chunk] = \
                    self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()

            # reset state before other computations
            self.reset_hidden()

            if loss_fcn != None and target_data != None:
                losses = loss_fcn(output_data, target_data)
                return input_data, target_data, output_data, losses

        return input_data, target_data, output_data

    def save_model(self,
                   file_name,
                   direc=''):
        if direc:
            utils.dir_check(direc)

        model_data = {'model_data': {'model_type': 'rnn',
                                     'input_size': self.input_size,
                                     'output_size': self.output_size,
                                     'unit_type': self.unit_type,
                                     'hidden_size': self.hidden_size,
                                     'skip': self.skip,
                                     'bias_fl': self.bias_fl,
                                     'nlayers': self.nlayers}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data['state_dict'] = model_state

        utils.json_save(model_data, file_name, direc)

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--input_size', type=int, default=1)
        parser.add_argument('--output_size', type=int, default=1)
        parser.add_argument('--unit_type', type=str, default="LSTM")
        parser.add_argument('--hidden_size', type=int, default=32)
        parser.add_argument('--skip', type=int, default=1)
        parser.add_argument('--bias_fl', default=True, action='store_true')
        parser.add_argument('--nlayers', type=int, default=1)

        return parser
