import argparse
import time
import os
import scipy
import torch
import torchinfo
import torch.utils.tensorboard as tensorboard

import src.utils as utils
import src.training as training
import src.data as data

from src.rnn import RNN
from src.gcn import GCN
from src.gcntfilm import GCNTF

torch.backends.cudnn.benchmark = True

train_configs = [
    {
        "name": "GCNTF1",
        "model_type": "gcntf",
        "nblocks": 1,
        "nlayers": 10,
        "nchannels": 16,
        "kernel_size": 3,
        "dilation_growth": 2,
        "tfilm_block_size": 128,
        "loss_fcns": {"L1": 0.5, "MSTFT": 0.5},
        "prefilt": None,
        "device": "fuzz-rndamp-G5S10A1msR2500ms",
        "file_name": "fuzz-rndamp-G5S10A1msR2500ms",
        "train_length": 112640,
        "val_chunk": 112640,
        "test_chunk": 112640,
        "validation_p": 20,
        "batch_size": 6
    },
    {
        "name": "GCNTF3",
        "model_type": "gcntf",
        "nblocks": 2,
        "nlayers": 9,
        "nchannels": 16,
        "kernel_size": 3,
        "dilation_growth": 2,
        "tfilm_block_size": 128,
        "loss_fcns": {"L1": 0.5, "MSTFT": 0.5},
        "prefilt": None,
        "device": "fuzz-rndamp-G5S10A1msR2500ms",
        "file_name": "fuzz-rndamp-G5S10A1msR2500ms",
        "train_length": 112640,
        "val_chunk": 112640,
        "test_chunk": 112640,
        "validation_p": 40,
        "batch_size": 6
    },
]

n_configs = len(train_configs)

for idx, tconf in enumerate(train_configs):

    parser = argparse.ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--model_type", type=str, default="gcntf", help="rnn, gcn, gcntf")
    # data locations, file names and config
    parser.add_argument("--device", "-d", default="ht1", help="device label")
    parser.add_argument("--data_rootdir", "-dr", default="./data", help="data directory")
    parser.add_argument("--file_name", "-fn", default="ht1", help="filename-input.wav and -target.wav")
    # parser.add_argument('--load_config', '-l', help="config file path")
    # parser.add_argument('--config_location', '-cl', default='configs', help='configs directory')
    parser.add_argument("--save_location", "-sloc", default="results", help="trained models directory")
    parser.add_argument("--load_model", "-lm", action="store_true", help="load pretrained model")

    # pre-processing of the training/val/test data
    parser.add_argument("--train_length", "-trlen", type=int, default=16384, help="training frame length in samples")
    parser.add_argument("--val_length", "-vllen", type=int, default=0, help="training frame length in samples")
    parser.add_argument("--test_length", "-tslen", type=int, default=0, help="training frame length in samples")

    # number of epochs and validation
    parser.add_argument("--epochs", "-eps", type=int, default=2000, help="max epochs")
    parser.add_argument("--validation_f", "-vfr", type=int, default=2, help="validation frequency (in epochs)")
    parser.add_argument("--validation_p", "-vp", type=int, default=25, help="validation patience or None")

    # settings for the training epoch
    parser.add_argument("--batch_size", "-bs", type=int, default=40, help="mini-batch size")
    parser.add_argument("--iter_num", "-it", type=int, default=None, help="batch_size set to have --iter_num batches")
    parser.add_argument("--learn_rate", "-lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--cuda", "-cu", default=1, help="use GPU if available")

    # loss function/s
    parser.add_argument("--loss_fcns", "-lf", default={"ESRPre": 0.75, "DC": 0.25}, help="loss functions and weights")
    parser.add_argument("--prefilt", "-pf", default="high_pass", help="pre-emphasis filter coefficients, can also read in a csv file")

    # validation and test sets chunk size
    parser.add_argument("--val_chunk", "-vs", type=int, default=100000, help="validation chunk length")
    parser.add_argument("--test_chunk", "-tc", type=int, default=100000, help="test chunk length")

    # parse general args
    args = parser.parse_args()

    # add model specific args
    if tconf["model_type"] == "rnn":
        parser = RNN.add_model_specific_args(parser)
    elif tconf["model_type"] == "gcn":
        parser = GCN.add_model_specific_args(parser)
    elif tconf["model_type"] == "gcntf":
        parser = GCNTF.add_model_specific_args(parser)

    # parse general + model args
    args = parser.parse_args()

    # create dictionary with args
    dict_args = vars(args)

    # overwrite with temporary configuration
    dict_args.update(tconf)

    # set filter args
    if dict_args["prefilt"] == "a-weighting":
        # as reported in in https://ieeexplore.ieee.org/abstract/document/9052944
        dict_args["prefilt"] = [0.85, 1]
    elif dict_args["prefilt"] == "high_pass":
        # args.prefilt = [-0.85, 1] # as reported in https://ieeexplore.ieee.org/abstract/document/9052944
        # as reported in (https://www.mdpi.com/2076-3417/10/3/766/htm)
        dict_args["prefilt"] = [-0.95, 1]
    else:
        dict_args["prefilt"] = None

    # directory where results will be saved
    if dict_args["model_type"] == "rnn":
        specifier = f"{idx+1}-{dict_args['name']}-{dict_args['device']}"
        specifier += f"__{dict_args['nlayers']}-{dict_args['hidden_size']}"
        specifier += "-skip" if dict_args["skip"] else "-noskip"
        specifier += f"__prefilt-{dict_args['prefilt']}-bs{dict_args['batch_size']}"
    elif dict_args["model_type"] == "gcn":
        specifier = f"{idx+1}-{dict_args['name']}-{dict_args['device']}"
        specifier += f"__{dict_args['nblocks']}-{dict_args['nlayers']}-{dict_args['nchannels']}"
        specifier += f"-{dict_args['kernel_size']}-{dict_args['dilation_growth']}"
        specifier += f"__prefilt-{dict_args['prefilt']}-bs{dict_args['batch_size']}"
    elif dict_args["model_type"] == "gcntf":
        specifier = f"{idx+1}-{dict_args['name']}-{dict_args['device']}"
        specifier += f"__{dict_args['nblocks']}-{dict_args['nlayers']}-{dict_args['nchannels']}"
        specifier += f"-{dict_args['kernel_size']}-{dict_args['dilation_growth']}-{dict_args['tfilm_block_size']}"
        specifier += f"__prefilt-{dict_args['prefilt']}-bs{dict_args['batch_size']}"

    # results directory
    save_path = os.path.join(dict_args["save_location"], specifier)
    utils.dir_check(save_path)

    # set the seed
    # TODO

    # init model
    if dict_args["model_type"] == "rnn":
        model = RNN(**dict_args)
    elif dict_args["model_type"] == "gcn":
        model = GCN(**dict_args)
    elif dict_args["model_type"] == "gcntf":
        model = GCNTF(**dict_args)

    # compute rf
    if dict_args["model_type"] in ["gcn", "gcntf"]:
        dict_args["rf"] = model.compute_receptive_field()
        
    model.save_state = False
    model.save_model("model", save_path)

    # save settings
    utils.json_save(dict_args, "config", save_path, indent=4)
    print(f"\n* Training config {idx+1}/{n_configs}")
    print(dict_args)

    # cuda
    if not torch.cuda.is_available() or dict_args["cuda"] == 0:
        print("\ncuda device not available/not selected")
        cuda = 0
    else:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.cuda.set_device(0)
        print("\ncuda device available")
        model = model.cuda()
        cuda = 1

    # optimiser + scheduler + loss fcns
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=dict_args["learn_rate"],
                                 weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                           "min",
                                                           factor=0.5,
                                                           patience=5,
                                                           verbose=True)

    loss_functions = training.LossWrapper(dict_args["loss_fcns"], dict_args["prefilt"])

    # training tracker
    train_track = training.TrainTrack()
    writer = tensorboard.SummaryWriter(os.path.join("results", specifier))

    # dataset
    dataset = data.DataSet(data_dir=dict_args["data_rootdir"])

    dataset.create_subset("train", frame_len=dict_args["train_length"])
    dataset.load_file(os.path.join("train", dict_args["file_name"]), "train")
    print("\ntrain dataset: ", dataset.subsets["train"].data["input"][0].shape)

    dataset.create_subset("val", frame_len=dict_args["val_length"])
    dataset.load_file(os.path.join("val", dict_args["file_name"]), "val")
    print("val dataset: ", dataset.subsets["val"].data["input"][0].shape)

    dataset.create_subset("test", frame_len=dict_args["test_length"])
    dataset.load_file(os.path.join("test", dict_args["file_name"]), "test")
    print("test dataset: ", dataset.subsets["test"].data["input"][0].shape)

    # summary
    print()
    torchinfo.summary(model,
                      input_size=(dict_args["train_length"], dict_args["batch_size"], 1),
                      device=None)

    # ===== TRAIN ===== #
    start_time = time.time()

    model.save_state = True
    patience_counter = 0
    init_time = 0

    for epoch in range(train_track['current_epoch'] + 1, dict_args["epochs"] + 1):
        ep_st_time = time.time()

        # run 1 epoch of training
        if dict_args["model_type"] == "rnn":
            epoch_losses = model.train_epoch(dataset.subsets['train'].data['input'][0],
                                             dataset.subsets['train'].data['target'][0],
                                             loss_functions,
                                             optimiser,
                                             dict_args["batch_size"],
                                             up_fr=dict_args["up_fr"])
        else:
            epoch_losses = model.train_epoch(dataset.subsets['train'].data['input'][0],
                                             dataset.subsets['train'].data['target'][0],
                                             loss_functions,
                                             optimiser,
                                             dict_args["batch_size"])
        epoch_loss = 0
        for loss in epoch_losses:
            epoch_loss += epoch_losses[loss]
        print(f"epoch {epoch} | \ttrain loss: \t{epoch_loss:0.4f}", end="")
        for loss in epoch_losses:
            print(f" | \t{loss}: \t{epoch_losses[loss]:0.4f}", end="")
        print()

        # ===== VALIDATION ===== #
        if epoch % dict_args["validation_f"] == 0:
            val_ep_st_time = time.time()
            val_input, val_target, val_output, val_losses = \
                model.process_data(dataset.subsets['val'].data['input'][0],
                                   dataset.subsets['val'].data['target'][0],
                                   loss_fcn=loss_functions,
                                   chunk=dict_args["val_chunk"])

            # val losses
            val_loss = 0
            for loss in val_losses:
                val_loss += val_losses[loss]
            print(f"\t\tval loss: \t{val_loss:0.4f}", end="")
            for loss in val_losses:
                print(f" | \t{loss}: \t{val_losses[loss]:0.4f}", end="")
            print()

            # update lr
            scheduler.step(val_loss)

            # save best model
            if val_loss < train_track['val_loss_best']:
                patience_counter = 0
                model.save_model('model_best', save_path)
                scipy.io.wavfile.write(os.path.join(save_path, "best_val_out.wav"),
                                       dataset.subsets['test'].fs,
                                       val_output.cpu().numpy()[:, 0, 0])
            else:
                patience_counter += 1

            # log validation losses
            for loss in val_losses:
                val_losses[loss] = val_losses[loss].item()

            train_track.val_epoch_update(val_loss=val_loss.item(),
                                         val_losses=val_losses,
                                         ep_st_time=val_ep_st_time,
                                         ep_end_time=time.time())

            writer.add_scalar('Loss/Val (Tot)', val_loss, epoch)
            for loss in val_losses:
                writer.add_scalar(f"Loss/Val ({loss})", val_losses[loss], epoch)

        # log training losses
        for loss in epoch_losses:
            epoch_losses[loss] = epoch_losses[loss].item()

        train_track.train_epoch_update(epoch_loss=epoch_loss.item(),
                                       epoch_losses=epoch_losses,
                                       ep_st_time=ep_st_time,
                                       ep_end_time=time.time(),
                                       init_time=init_time,
                                       current_ep=epoch)

        writer.add_scalar('Loss/Train (Tot)', epoch_loss, epoch)
        for loss in epoch_losses:
            writer.add_scalar(f"Loss/Train ({loss})", epoch_losses[loss], epoch)

        # log learning rate
        writer.add_scalar('LR/current', optimiser.param_groups[0]['lr'])

        # save model
        model.save_model('model', save_path)

        # log training stats to json
        utils.json_save(train_track, 'training_stats', save_path, indent=4)

        # check early stopping
        if dict_args["validation_p"] and patience_counter > dict_args["validation_p"]:
            print('\nvalidation patience limit reached at epoch ' + str(epoch))
            break

    # ===== TEST (last model) ===== #
    test_input, test_target, test_output, test_losses = \
        model.process_data(dataset.subsets['test'].data['input'][0],
                           dataset.subsets['test'].data['target'][0],
                           loss_fcn=loss_functions,
                           chunk=dict_args["test_chunk"])

    # test losses (last)
    test_loss = 0
    for loss in test_losses:
        test_loss += test_losses[loss]
    print(f"\ttest loss (last): \t{test_loss:0.4f}", end="")
    for loss in test_losses:
        print(f" | \t{loss}: \t{test_losses[loss]:0.4f}", end="")
    print()

    lossESR = training.ESRLoss()  # include ESR loss
    test_loss_ESR = lossESR(test_output, test_target)

    # save output audio
    scipy.io.wavfile.write(os.path.join(save_path, "test_out_final.wav"),
                           dataset.subsets['test'].fs,
                           test_output.cpu().numpy()[:, 0, 0])

    # log test losses
    for loss in test_losses:
        test_losses[loss] = test_losses[loss].item()

    train_track['test_loss_final'] = test_loss.item()
    train_track['test_losses_final'] = test_losses
    train_track['test_lossESR_final'] = test_loss_ESR.item()

    writer.add_scalar('Loss/Test/Last (Tot)', test_loss, 0)
    for loss in test_losses:
        writer.add_scalar(f"Loss/Test/Last ({loss})", test_losses[loss], 0)

    # ===== TEST (best validation model) ===== #
    best_val_net = utils.json_load('model_best', save_path)
    model = utils.load_model(best_val_net)

    test_input, test_target, test_output, test_losses = \
        model.process_data(dataset.subsets['test'].data['input'][0],
                           dataset.subsets['test'].data['target'][0],
                           loss_fcn=loss_functions,
                           chunk=dict_args["test_chunk"])

    # test losses (best)
    test_loss = 0
    for loss in test_losses:
        test_loss += test_losses[loss]
    print(f"\ttest loss (best): \t{test_loss:0.4f}", end="")
    for loss in test_losses:
        print(f" | \t{loss}: \t{test_losses[loss]:0.4f}", end="")
    print()

    test_loss_ESR = lossESR(test_output, test_target)

    # save output audio
    scipy.io.wavfile.write(os.path.join(save_path, "test_out_bestv.wav"),
                           dataset.subsets['test'].fs,
                           test_output.cpu().numpy()[:, 0, 0])

    # log test losses
    for loss in test_losses:
        test_losses[loss] = test_losses[loss].item()

    train_track['test_loss_best'] = test_loss.item()
    train_track['test_losses_best'] = test_losses
    train_track['test_lossESR_best'] = test_loss_ESR.item()

    writer.add_scalar('Loss/Test/Best (Tot)', test_loss, 0)
    for loss in test_losses:
        writer.add_scalar(f"Loss/Test/Best ({loss})", test_losses[loss], 0)

    # log training stats to json
    utils.json_save(train_track, 'training_stats', save_path, indent=4)

    if cuda:
        with open(os.path.join(save_path, 'maxmemusage.txt'), 'w') as f:
            f.write(str(torch.cuda.max_memory_allocated()))

    stop_time = time.time()
    print(f"\ntraining time: {(stop_time-start_time)/60:0.2f} min")
