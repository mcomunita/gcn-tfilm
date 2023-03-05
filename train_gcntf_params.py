import argparse
import time
import os
import scipy
import numpy as np
import torch
import torchinfo
import torch.utils.tensorboard as tensorboard

from re import split as resplit

import src.utils as utils
import src.training as training

from src.dataset import FuzzDataset
from src.gcntfilm import GCNTF

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

train_configs = [
    {
        "name": "GCNTF3",
        "model_type": "gcntf",
        "nblocks": 1,
        "nlayers": 10,
        "nchannels": 16,
        "kernel_size": 3,
        "dilation_growth": 2,
        "tfilm_block_size": 128,

        "loss_fcns": {"L1": 0.5, "MSTFT": 0.5},
        "prefilt": None,
        
        "label": "fuzz",
        "data_rootdir": "data/fuzz",
        "nparams": 4,
        "sample_length": 48000*5,
        "preload": False,
        "data_split": [.08, .01, .01],
        "num_workers": 0,

        "epochs": 2,
        "validation_f": 2,
        "scheduler_p": 20,
        "validation_p": 40,
        "batch_size": 6,
        "learn_rate": 5e-3,

        "cuda": 1
    },
]


n_configs = len(train_configs)


for idx, tconf in enumerate(train_configs):

    parser = argparse.ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--model_type", type=str, default="gcntf", help="gcntf, rnn")

    # data locations, file names and config
    parser.add_argument("--label", help="device label")
    parser.add_argument("--data_rootdir", default="./data", help="data directory")
    parser.add_argument("--params_file", default="./data/params.csv", help="parameters directory")

    parser.add_argument("--save_location", default="results", help="trained models directory")
    parser.add_argument("--load_model", action="store_true", help="load pretrained model")

    # dataset
    parser.add_argument("--sample_length", type=int, default=48000, help="training frame length in samples")
    parser.add_argument('--num_workers', type=int, default=16)

    # number of epochs and validation
    parser.add_argument("--epochs", type=int, default=2000, help="max epochs")
    parser.add_argument("--validation_f", type=int, default=2, help="validation frequency (in epochs)")
    parser.add_argument("--validation_p", type=int, default=25, help="validation patience or None")

    # settings for the training epoch
    parser.add_argument("--batch_size", type=int, default=40, help="mini-batch size")
    parser.add_argument("--learn_rate", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--cuda", default=1, help="use GPU if available")

    # loss function/s
    parser.add_argument("--loss_fcns", default={"ESRPre": 0.75, "DC": 0.25}, help="loss functions and weights")
    parser.add_argument("--prefilt", default="high_pass", help="pre-emphasis filter coefficients, can also read in a csv file")

    # parse general args
    args = parser.parse_args()

    # add model specific args
    if tconf["model_type"] == "gcntf":
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
    if dict_args["model_type"] == "gcntf":
        specifier = f"{idx+1}-{dict_args['name']}-{dict_args['label']}"
        specifier += f"__{dict_args['nblocks']}-{dict_args['nlayers']}-{dict_args['nchannels']}"
        specifier += f"-{dict_args['kernel_size']}-{dict_args['dilation_growth']}-{dict_args['tfilm_block_size']}"
        specifier += f"__prefilt-{dict_args['prefilt']}-bs{dict_args['batch_size']}"

    # results directory
    save_path = os.path.join(dict_args["save_location"], specifier)
    valbest_out_path = os.path.join(save_path, "val_best_out")
    test_final_out_path = os.path.join(save_path, "test_final_out")
    test_valbest_out_path = os.path.join(save_path, "test_bestval_out")
    utils.dir_check(dict_args["save_location"])
    utils.dir_check(save_path)
    utils.dir_check(valbest_out_path)
    utils.dir_check(test_final_out_path)
    utils.dir_check(test_valbest_out_path)

    print()
    print(save_path)
    print(valbest_out_path)
    print(test_final_out_path)
    print(test_valbest_out_path)

    # set the seed
    # TODO

    # cuda
    if not torch.cuda.is_available() or dict_args["cuda"] == 0:
        torch.set_default_tensor_type("torch.FloatTensor")
        device = torch.device("cpu")
        print("\ncuda device not available/not selected")
    else:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("\ncuda device available")

    # init model
    if dict_args["model_type"] == "gcntf":
        model = GCNTF(**dict_args, device=device)

    # compute rf
    dict_args["rf"] = model.compute_receptive_field()

    model.save_state = False
    model.save_model('model', save_path)

    # save settings
    utils.json_save(dict_args, "config", save_path, indent=4)
    print(f"\n* Training config {idx+1}/{n_configs}")
    print(dict_args)

    # optimiser + scheduler + loss fcns
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=dict_args["learn_rate"],
                                 weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                           "min",
                                                           factor=0.5,
                                                           patience=dict_args["scheduler_p"],
                                                           verbose=True)

    loss_functions = training.LossWrapper(dict_args["loss_fcns"], dict_args["prefilt"])

    # training tracker
    train_track = training.TrainTrack()
    writer = tensorboard.SummaryWriter(os.path.join("results", specifier))

    # dataset
    print("\n== DATASET ==")
    dataset = FuzzDataset(root_dir=dict_args["data_rootdir"],
                          sample_length=dict_args["sample_length"],
                          preload=dict_args["preload"])

    split = np.array(dict_args["data_split"])
    split_lengths = np.int_(split * len(dataset))
    split_idxs = [split_lengths[0], sum(split_lengths[:2]), sum(split_lengths)]

    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)

    train_idxs = idxs[:split_idxs[0]]
    val_idxs = idxs[split_idxs[0]:split_idxs[1]]
    test_idxs = idxs[split_idxs[1]:split_idxs[2]]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idxs)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idxs)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=dict_args["batch_size"],
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   num_workers=dict_args["num_workers"])
    val_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=dict_args["batch_size"],
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=dict_args["num_workers"])
    test_dataloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=dict_args["batch_size"],
                                                  sampler=test_sampler,
                                                  shuffle=False,
                                                  num_workers=dict_args["num_workers"])
    print()
    print("\nfull dataset: ", len(dataset))
    print("train dataset: ", len(train_idxs))
    print("val dataset: ", len(val_idxs))
    print("test dataset: ", len(test_idxs))

    # summary
    print()
    torchinfo.summary(model,
                      input_size=[(dict_args["batch_size"], 1, dict_args["rf"]),
                                  (dict_args["batch_size"], dict_args["nparams"])],
                      col_names=["input_size", "output_size", "num_params"],
                      device=device)
    
    # ===== TRAIN ===== #
    print("\n== TRAIN ==")
    model = model.to(device)

    start_time = time.time()

    model.save_state = True
    patience_counter = 0
    init_time = 0

    for epoch in range(train_track['current_epoch'] + 1, dict_args["epochs"] + 1):
        ep_st_time = time.time()

        # run 1 epoch of training
        epoch_losses = model.train_epoch(train_dataloader,
                                         loss_functions,
                                         optimiser)
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
            val_losses = model.val_epoch(val_dataloader,
                                         loss_functions)

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

                # save data
                os.system(f"rm -rf {valbest_out_path}/*") # delete previous files
                for idx in val_idxs:
                    sample = dataset.getsample(idx)
                    # process
                    val_output = model.process_data(input=sample["input_audio"].unsqueeze(0),
                                                    params=sample["params"].unsqueeze(0))
                    # filenames
                    offset_in_seconds = sample["offset"] // sample["sr"]
                    ifile = os.path.basename(sample["input_file"])[:-4]
                    ifile = f"{ifile}_{offset_in_seconds}.wav"
                    ofile = resplit("_", os.path.basename(sample["target_file"])[:-4])
                    tfile = f"{ofile[:1]}_target_{ofile[3]}_{offset_in_seconds}.wav"
                    pfile = f"{ofile[:1]}_pred_{ofile[3]}_{offset_in_seconds}.wav"

                    # save
                    scipy.io.wavfile.write(os.path.join(valbest_out_path, ifile),
                                           sample["sr"],
                                           sample["input_audio"].numpy()[0, :])
                    scipy.io.wavfile.write(os.path.join(valbest_out_path, tfile),
                                           sample["sr"],
                                           sample["target_audio"].numpy()[0, :])
                    scipy.io.wavfile.write(os.path.join(valbest_out_path, pfile),
                                           sample["sr"],
                                           val_output.cpu().numpy()[0, 0, :])
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
                writer.add_scalar(
                    f"Loss/Val ({loss})", val_losses[loss], epoch)

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
            writer.add_scalar(
                f"Loss/Train ({loss})", epoch_losses[loss], epoch)

        # log learning rate
        writer.add_scalar('LR/current', optimiser.param_groups[0]['lr'])

        # save model
        model.save_model('model', save_path)

        # log training stats to json
        utils.json_save(train_track, 'training_stats', save_path, indent=4)

        # check early stopping
        if dict_args["validation_p"] and patience_counter > dict_args["validation_p"]:
            print("early stop: ", patience_counter)
            print('\nvalidation patience limit reached at epoch ' + str(epoch))
            break

    # ===== TEST (last model) ===== #
    print("\n== TEST (last) ==")
    test_losses = model.test_epoch(test_dataloader,
                                   loss_functions)

    test_loss = 0
    for loss in test_losses:
        test_loss += test_losses[loss]
    print(f"\ttest loss (last): \t{test_loss:0.4f}", end="")
    for loss in test_losses:
        print(f" | \t{loss}: \t{test_losses[loss]:0.4f}", end="")
    print()

    # save data
    for idx in test_idxs:
        sample = dataset.getsample(idx)
        # process
        test_output = model.process_data(input=sample["input_audio"].unsqueeze(0),
                                         params=sample["params"].unsqueeze(0))
        # filenames
        offset_in_seconds = sample["offset"] // sample["sr"]
        ifile = os.path.basename(sample["input_file"])[:-4]
        ifile = f"{ifile}_{offset_in_seconds}.wav"
        ofile = resplit("_", os.path.basename(sample["target_file"])[:-4])
        tfile = f"{ofile[:1]}_target_{ofile[3]}_{offset_in_seconds}.wav"
        pfile = f"{ofile[:1]}_pred_{ofile[3]}_{offset_in_seconds}.wav"

        # save
        scipy.io.wavfile.write(os.path.join(test_final_out_path, ifile),
                               sample["sr"],
                               sample["input_audio"].numpy()[0, :])
        scipy.io.wavfile.write(os.path.join(test_final_out_path, tfile),
                               sample["sr"],
                               sample["target_audio"].numpy()[0, :])
        scipy.io.wavfile.write(os.path.join(test_final_out_path, pfile),
                               sample["sr"],
                               test_output.cpu().numpy()[0, 0, :])

    # log test losses
    for loss in test_losses:
        test_losses[loss] = test_losses[loss].item()

    train_track['test_loss_final'] = test_loss.item()
    train_track['test_losses_final'] = test_losses

    writer.add_scalar('Loss/Test/Last (Tot)', test_loss, 0)
    for loss in test_losses:
        writer.add_scalar(f"Loss/Test/Last ({loss})", test_losses[loss], 0)

    # ===== TEST (best validation model) ===== #
    print("\n== TEST (best) ==")
    best_val_net = utils.json_load('model_best', save_path)
    model = utils.load_model(best_val_net, device=device)
    model.device = device
    model = model.to(device)

    test_losses = model.test_epoch(test_dataloader,
                                   loss_functions)

    # test losses (best)
    test_loss = 0
    for loss in test_losses:
        test_loss += test_losses[loss]
    print(f"\ttest loss (best): \t{test_loss:0.4f}", end="")
    for loss in test_losses:
        print(f" | \t{loss}: \t{test_losses[loss]:0.4f}", end="")
    print()

    # save data
    for idx in test_idxs:
        sample = dataset.getsample(idx)
        # process
        test_output = model.process_data(input=sample["input_audio"].unsqueeze(0),
                                         params=sample["params"].unsqueeze(0))
        # filenames
        offset_in_seconds = sample["offset"] // sample["sr"]
        ifile = os.path.basename(sample["input_file"])[:-4]
        ifile = f"{ifile}_{offset_in_seconds}.wav"
        ofile = resplit("_", os.path.basename(sample["target_file"])[:-4])
        tfile = f"{ofile[:1]}_target_{ofile[3]}_{offset_in_seconds}.wav"
        pfile = f"{ofile[:1]}_pred_{ofile[3]}_{offset_in_seconds}.wav"

        # save
        scipy.io.wavfile.write(os.path.join(test_valbest_out_path, ifile),
                               sample["sr"],
                               sample["input_audio"].numpy()[0, :])
        scipy.io.wavfile.write(os.path.join(test_valbest_out_path, tfile),
                               sample["sr"],
                               sample["target_audio"].numpy()[0, :])
        scipy.io.wavfile.write(os.path.join(test_valbest_out_path, pfile),
                               sample["sr"],
                               test_output.cpu().numpy()[0, 0, :])

    # log test losses
    for loss in test_losses:
        test_losses[loss] = test_losses[loss].item()

    train_track['test_loss_best'] = test_loss.item()
    train_track['test_losses_best'] = test_losses

    writer.add_scalar('Loss/Test/Best (Tot)', test_loss, 0)
    for loss in test_losses:
        writer.add_scalar(f"Loss/Test/Best ({loss})", test_losses[loss], 0)

    # log training stats to json
    utils.json_save(train_track, 'training_stats', save_path, indent=4)

    # if cuda:
    #     with open(os.path.join(save_path, 'maxmemusage.txt'), 'w') as f:
    #         f.write(str(torch.cuda.max_memory_allocated()))

    stop_time = time.time()
    print(f"\ntraining time: {(stop_time-start_time)/60:0.2f} min")
