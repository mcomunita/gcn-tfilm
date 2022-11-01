import os
import json
import torch
from src.rnn import RNN
from src.gcn import GCN
from src.gcntfilm import GCNTF


# Function that checks if a directory exists, and creates it if it doesn't, if dir_name is a list of strings, it will
# create a search path, i.e dir_name = ['directory', 'subdir'] will search for directory 'directory/subdir'
def dir_check(dir_name):
    dir_name = [dir_name] if not type(dir_name) == list else dir_name
    dir_path = os.path.join(*dir_name)
    if os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)


# Function that takes a file_name and optionally a path to the directory the file is expected to be, returns true if
# the file is found in the stated directory (or the current directory is dir_name = '') or False is dir/file isn't found
def file_check(file_name, dir_name=''):
    assert type(file_name) == str
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    full_path = os.path.join(*dir_name, file_name)
    return os.path.isfile(full_path)


# Function that saves 'data' to a json file. Constructs a file path is dir_name is provided.
def json_save(data, file_name, dir_name='', indent=0):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    assert type(file_name) == str
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path, 'w') as fp:
        json.dump(data, fp, indent=indent)


def json_load(file_name, dir_name=''):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path) as fp:
        return json.load(fp)


def load_model(model_data):
    model_meta = model_data.pop('model_data')
    # print(model_meta)

    if model_meta["model_type"] == "rnn":
        model = RNN(**model_meta)
    elif model_meta["model_type"] == "gcn":
        model = GCN(**model_meta)
    elif model_meta["model_type"] == "gcntf":
        model = GCNTF(**model_meta)

    if 'state_dict' in model_data:
        state_dict = model.state_dict()
        for each in model_data['state_dict']:
            state_dict[each] = torch.tensor(model_data['state_dict'][each])
        model.load_state_dict(state_dict)

    return model
