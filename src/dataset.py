import os
import sys
import glob
import torch 
import torchaudio
import numpy as np

from re import split as resplit
from natsort import natsorted


def rand(low=0, high=1):
    return (torch.rand(1).numpy()[0] * (high - low)) + low


def randint(low=0, high=1):
    return torch.randint(low, high + 1, (1,)).numpy()[0]

class FuzzDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 sample_length=48000,
                 preload=False):
        super().__init__()
        self.root_dir = root_dir
        self.sample_length = sample_length
        self.preload = preload

        # get file paths
        self.input_files = natsorted(glob.glob(os.path.join(self.root_dir, "*input*.wav")))
        self.target_files = natsorted(glob.glob(os.path.join(self.root_dir, "*target*.wav")))

        self.n_inputs = len(self.input_files)
        self.n_targets = len(self.target_files)

        # get audio samples and params
        self.samples = [] 
        self.tot_frames = 0  # total number of frames in dataset
        
        # loop target files to count total length
        print()
        for iidx, ifile in enumerate(self.input_files):
            # print(ifile)
            if self.preload:
                input, sr = torchaudio.load(ifile, normalize=False)

            for tidx, tfile in enumerate(self.target_files):
                # print(tfile)

                md = torchaudio.info(tfile)
                num_frames = md.num_frames # num samples
                self.tot_frames += num_frames

                params = resplit("_|.wav", os.path.basename(tfile))[3] # get params string
                params = resplit("-", params) # split params string
                params = [p[1:] for p in params] # remove control letters
                params = [float(p)/100 for p in params] # convert into [0,1]
                params = torch.tensor(params) # tensor

                if self.preload:
                    sys.stdout.write(f"* Pre-loading... {tidx+1:3d}/{len(self.target_files):3d} ...\r")
                    sys.stdout.flush()
                    
                    target, sr = torchaudio.load(tfile, normalize=False)
                    num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                else:
                    input = None
                    target = None
                    sr = None

                # create one entry for each file
                self.file_samples = []
                if self.sample_length == -1: # take whole file
                    self.file_samples.append({"iidx": iidx,
                                              "tidx": tidx, 
                                              "input_file" : ifile,
                                              "target_file" : tfile,
                                              "input_audio" : input if input is not None else None,
                                              "target_audio" : target if input is not None else None,
                                              "params" : params,
                                              "offset": 0,
                                              "frames" : num_frames,
                                              "sr": sr})
                else: # split into chunks
                    for n in range((num_frames // self.sample_length)):
                        offset = int(n * self.sample_length)
                        end = offset + self.sample_length
                        self.file_samples.append({"iidx": iidx,
                                                  "tidx": tidx,
                                                  "input_file" : ifile,
                                                  "target_file" : tfile,
                                                  "input_audio" : input[:,offset:end] if input is not None else None,
                                                  "target_audio" : target[:,offset:end] if input is not None else None,
                                                  "params" : params,
                                                  "offset": offset,
                                                  "frames" : num_frames,
                                                  "sr": sr})

                # add to overall file examples
                self.samples += self.file_samples

            print(f"total minutes: {self.tot_frames/48000//60}")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        if self.preload:
            input = self.samples[idx]["input_audio"]
            target = self.samples[idx]["target_audio"]
        else:
            if self.sample_length == -1: # whole file
                input, sr  = torchaudio.load(self.samples[idx]["input_file"], 
                                            normalize=False)
                target, sr = torchaudio.load(self.samples[idx]["target_file"], 
                                            normalize=False)
            else:
                offset = self.samples[idx]["offset"]
                input, sr  = torchaudio.load(self.samples[idx]["input_file"], 
                                            num_frames=self.sample_length, 
                                            frame_offset=offset, 
                                            normalize=False)
                target, sr = torchaudio.load(self.samples[idx]["target_file"], 
                                            num_frames=self.sample_length, 
                                            frame_offset=offset, 
                                            normalize=False)
        # then get the tuple of parameters
        params = self.samples[idx]["params"]

        return input, target, params
    
    
    def getsample(self, idx):
        if self.preload:
            return self.samples[idx]
        else:
            if self.sample_length == -1: # whole file
                input, sr  = torchaudio.load(self.samples[idx]["input_file"], 
                                            normalize=False)
                target, sr = torchaudio.load(self.samples[idx]["target_file"], 
                                            normalize=False)
            else:
                offset = self.samples[idx]["offset"]
                input, sr  = torchaudio.load(self.samples[idx]["input_file"], 
                                            num_frames=self.sample_length, 
                                            frame_offset=offset, 
                                            normalize=False)
                target, sr = torchaudio.load(self.samples[idx]["target_file"], 
                                            num_frames=self.sample_length, 
                                            frame_offset=offset, 
                                            normalize=False)
        sample = self.samples[idx]
        sample["input_audio"] = input
        sample["target_audio"] = target
        sample["sr"] = sr

        return sample