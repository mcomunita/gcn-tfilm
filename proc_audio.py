import argparse
import torch

from scipy.io.wavfile import write

import src.utils as utils
import src.data as data


def parse_args():
    parser = argparse.ArgumentParser(
        description='''This script takes an input .wav file, loads it and processes it with a neural network model of a
                    device, i.e guitar amp/pedal, and saves the output as a .wav file''')

    parser.add_argument('--input_file', type=str, default='data/test/ht1-input.wav', help='input file')
    parser.add_argument('--output_file', type=str, default='output.wav', help='output file')
    parser.add_argument('--model_file', type=str, default='results/ht1-ht11/model_best.json', help='model file')
    parser.add_argument('--chunk_length', type=int, default=16384, help='chunk length')
    return parser.parse_args()


def proc_audio(args):
    model_data = utils.json_load(args.model_file)
    model = utils.load_model(model_data)

    dataset = data.DataSet(data_dir='', extensions='')
    dataset.create_subset('data')
    dataset.load_file(args.input_file, set_names='data')

    # cuda
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # torch.cuda.set_device(0)
        print("\ncuda device available")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
        print("\ncuda device NOT available")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_data = dataset.subsets['data'].data['data'][0]

        input_data = input_data.to(device)
        # output = network(input_data)

        _, _, output = model.process_data(input_data,
                                          chunk=args.chunk_length)

    write(args.output_file, dataset.subsets['data'].fs, output.cpu().numpy()[:, 0, 0])


def main():
    args = parse_args()
    proc_audio(args)


if __name__ == '__main__':
    main()
