import torch
import torch.nn as nn
import auraloss

# ESR loss calculates the Error-to-signal between the output/target


class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


# PreEmph is a class that applies an FIR pre-emphasis filter to the signal, the filter coefficients are in the
# filter_cfs argument, and lp is a flag that also applies a low pass filter
# Only supported for single-channel!
class PreEmph(nn.Module):
    def __init__(self, filter_cfs, low_pass=0):
        super(PreEmph, self).__init__()
        self.epsilon = 0.00001
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = nn.Conv1d(1, 1, 2, bias=False)
        self.conv_filter.weight.data = torch.tensor([[filter_cfs]], requires_grad=False)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = nn.Conv1d(1, 1, 2, bias=False)
            self.lp_filter.weight.data = torch.tensor([[[0.85, 1]]], requires_grad=False)

    def forward(self, output, target):
        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(self.zPad, output.shape[1], 1), output))
        target = torch.cat((torch.zeros(self.zPad, target.shape[1], 1), target))
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)


class LossWrapper(nn.Module):
    def __init__(self, losses, pre_filt=None):
        super(LossWrapper, self).__init__()
        self.losses = losses
        self.loss_dict = {
            'ESR': ESRLoss(),
            'DC': DCLoss(),
            'L1': torch.nn.L1Loss(),
            'STFT': auraloss.freq.STFTLoss(),
            'MSTFT': auraloss.freq.MultiResolutionSTFTLoss()
        }
        if pre_filt:
            pre_filt = PreEmph(pre_filt)
            self.loss_dict['ESRPre'] = lambda output, target: self.loss_dict['ESR'].forward(*pre_filt(output, target))
        loss_functions = [[self.loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(torch.Tensor([items[1] for items in loss_functions]))
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output, target):
        all_losses = {}
        for i, loss in enumerate(self.losses):
            # original shape: length x batch x 1
            # auraloss needs: batch x 1 x length
            loss_fcn = self.loss_functions[i]
            loss_factor = self.loss_factors[i]
            # if isinstance(loss_fcn, auraloss.freq.STFTLoss) or isinstance(loss_fcn, auraloss.freq.MultiResolutionSTFTLoss):
            #     output = torch.permute(output, (1, 2, 0))
            #     target = torch.permute(target, (1, 2, 0))
            all_losses[loss] = torch.mul(loss_fcn(output, target), loss_factor)
        return all_losses


class TrainTrack(dict):
    def __init__(self):
        self.update({'current_epoch': 0,
                     
                     'tot_train_losses': [],
                     'train_losses': [], 
                     
                     'tot_val_losses': [],
                     'val_losses': [],
                     
                     'train_av_time': 0.0,
                     'val_av_time': 0.0, 
                     'total_time': 0.0, 
                     
                     'val_loss_best': 1e12,
                     'val_losses_best': 1e12,

                     'test_loss_final': 0,
                     'test_losses_final': {},

                     'test_loss_best': 0,
                     'test_losses_best': {}})

    def restore_data(self, training_info):
        self.update(training_info)

    def train_epoch_update(self, epoch_loss, epoch_losses, ep_st_time, ep_end_time, init_time, current_ep):
        self['current_epoch'] = current_ep
        self['tot_train_losses'].append(epoch_loss)
        self['train_losses'].append(epoch_losses)
        
        if self['train_av_time']:
            self['train_av_time'] = (self['train_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['train_av_time'] = ep_end_time - ep_st_time
        
        self['total_time'] += ((init_time + ep_end_time - ep_st_time)/3600)

    def val_epoch_update(self, val_loss, val_losses, ep_st_time, ep_end_time):
        self['tot_val_losses'].append(val_loss)
        self['val_losses'].append(val_losses)
        
        if self['val_av_time']:
            self['val_av_time'] = (self['val_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['val_av_time'] = ep_end_time - ep_st_time

        if val_loss < self['val_loss_best']:
            self['val_loss_best'] = val_loss
            self['val_losses_best'] = val_losses
