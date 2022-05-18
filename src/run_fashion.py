import filelock
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time
import traceback
import torch
import torch.utils.data as tud
from torchvision import datasets
import torchvision.transforms as T
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm

from EinsumNetwork import Graph, EinsumNetwork
import utils


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class UnsupervisedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        datapoint = self.base[idx]
        if isinstance(datapoint, tuple):
            datapoint, label = datapoint
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint

    

dataset = FashionMNIST

n_epochs = 300
batch_size = 512
latent_dim = 16
n_filters = 16
height = 28
width = 28

if dataset in [MNIST, FashionMNIST]:
    transf = T.Compose(
        [T.ToTensor()]
    )

    train = UnsupervisedDataset(dataset(root='../../data', train=True, download=True, transform=transf))
    train, valid = torch.utils.data.random_split(train, [50000, 10000])
    test = UnsupervisedDataset(dataset(root='../../data', train=False, download=True, transform=transf))

    batch_size = 128
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)


training = True
device = 'cuda'

result_base_path = '../results/einet/fashion-mnist/'

# structure_range = ['poon_domingos', 'poon_domingos_vertical', 'poon_domingos_horizontal', 'vtree_vertical', 'vtree_horizontal']
structure = 'poon_domingos' # ['poon_domingos_vertical', 'poon_domingos_horizontal']

poon_domingos_pieces = [7] # [[4], [7], [4, 16]]
poon_domingos_1d_pieces = [4] # [[4], [8], [16]]
vtree_delta = [8]
vtree_mode = 'balanced' # 'sliced'

num_sums = 30 # [10, 20, 30, 40]

exponential_family = EinsumNetwork.NormalArray
exponential_family_args = {'min_var': 1., 'max_var': 1.}

num_epochs = 300
online_em_frequency = 50
online_em_stepsize = 0.5

early_stopping_epochs = 30
warmup = 100

train_time_limit = 3600 * 6
worker_time_limit = 42600
worker_start = time.time()


def make_shuffled_batch(N, batch_size):
    idx = np.random.permutation(N)
    num_full_batches = N // batch_size
    k = num_full_batches * batch_size
    b_idx = np.array_split(idx[0:k], num_full_batches)
    if k < N:
        b_idx.append(idx[k:])
    return b_idx


def eval_ll(spn, valid_x, batch_size):
    with torch.no_grad():
        ll = 0.0
        for batch in valid_x:
            batch = batch.permute(0, 2, 3, 1)
            batch = batch.to(torch.float).to(device)
            batch = batch.reshape(batch.shape[0], height * width, 1)
            ll_sample = spn.forward(batch)
            ll = ll_sample.sum() + ll
        return ll / len(valid_x.dataset)


def load(spn, result_path):
    model_file = os.path.join(result_path, 'einet.mdl')
    record_file = os.path.join(result_path, 'record.pkl')
    sample_dir = os.path.join(result_path, 'samples')
    utils.mkdir_p(sample_dir)

    if os.path.isfile(model_file) and os.path.isfile(record_file):
        spn.load_state_dict(torch.load(model_file))
        record = pickle.load(open(record_file, 'rb'))
        print("Loaded model")
        return spn
    else:
        print('Model not found')
        return None


def train(spn, train_x, valid_x, test_x, result_path, train_time_limit, worker_time_limit):

    e = 0
    enter_time = time.time()
    time_mark = time.time()

    model_file = os.path.join(result_path, 'einet.mdl')
    record_file = os.path.join(result_path, 'record.pkl')
    sample_dir = os.path.join(result_path, 'samples')
    utils.mkdir_p(sample_dir)

    if os.path.isfile(model_file) and os.path.isfile(record_file):
        spn.load_state_dict(torch.load(model_file))
        record = pickle.load(open(record_file, 'rb'))
        print("Loaded model")
    else:
        record = {'valid_ll': [],
                  'test_ll': [],
                  'epoch_count': 0,
                  'epoch_times': [],
                  'elapsed_time': 0.0,
                  'best_validation_ll': None}

    for _epc, epoch_count in enumerate(range(record['epoch_count'], num_epochs)):

        epoch_start = time.time()

        for batch in tqdm(train_x):
            batch = batch.permute(0, 2, 3, 1)
            batch = batch.to(torch.float).to(device)
            batch = batch.reshape(batch.shape[0], height * width, 1)

            ll_sample = spn.forward(batch)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()
            spn.em_process_batch()
        spn.em_update()

        epoch_time = time.time() - epoch_start

        ##### evaluate
        valid_ll = eval_ll(spn, valid_x, batch_size=batch_size)
        test_ll = eval_ll(spn, test_x, batch_size=batch_size)

        ##### store results
        record['valid_ll'].append(valid_ll)
        record['test_ll'].append(test_ll)
        record['epoch_count'] = epoch_count + 1
        record['epoch_times'].append(epoch_time)
        record['elapsed_time'] += time.time() - time_mark
        time_mark = time.time()

        pickle.dump(record, open(record_file, 'wb'))

        print("[{}, {}]   valid LL {}   test LL {}".format(
            epoch_count,
            record['elapsed_time'],
            valid_ll,
            test_ll))

        if record['best_validation_ll'] is None or valid_ll > record['best_validation_ll']:
            record['best_validation_ll'] = valid_ll
            torch.save(spn.state_dict(), model_file)
            e = 0
        else:
            e = e + 1
        if epoch_count < warmup:
            e = 0
        if e > early_stopping_epochs:
            print('Early stopping --- break')
            break

        ##### check if enough training time
        if record['elapsed_time'] > train_time_limit:
            print("train timeout --- break")
            return None

        ##### check if enough worker time
        elapsed_time = time.time() - enter_time
        remaining_time = worker_time_limit - elapsed_time
        if remaining_time < 1.05 * (elapsed_time / (_epc + 1)):
            print("short of worker time --- break")
            return -1

if structure == 'poon_domingos':
    structure_param = poon_domingos_pieces
elif structure == 'vtree_vertical' or structure == 'vtree_horizontal':
    structure_param = vtree_delta
elif structure == 'poon_domingos_vertical' or structure == 'poon_domingos_horizontal':
    structure_param = poon_domingos_1d_pieces
else:
    raise AssertionError

                   
result_path = result_base_path
if structure == 'poon_domingos':
    result_path = os.path.join(result_path, "poon-domingos_" + "_".join([str(i) for i in structure_param]))
    pd_delta = [[height / d, width / d] for d in structure_param]
    rg = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
elif structure == 'vtree_vertical' or structure == 'vtree_horizontal':
    result_path = os.path.join(result_path, structure + '_' + vtree_mode + '_{}'.format(structure_param))
    axis = 1 if structure == 'vtree_vertical' else 0
    if vtree_mode == 'sliced':
        rg = Graph.vtree_sliced_structure(shape=(height, width), axis=axis, delta=structure_param)
    elif vtree_mode == 'balanced':
        rg = Graph.vtree_balanced_structure(shape=(height, width), axis=axis, delta=structure_param)
    else:
        raise AssertionError
elif structure == 'poon_domingos_vertical' or structure == 'poon_domingos_horizontal':
    result_path = os.path.join(result_path, structure + '_' + "_".join([str(i) for i in structure_param]))
    if structure == 'poon_domingos_vertical':
        pd_delta = [[width / d] for d in structure_param]
        axes = [1]
    else:
        pd_delta = [[height / d] for d in structure_param]
        axes = [0]
    rg = Graph.poon_domingos_structure(shape=(height, width), axes=axes, delta=pd_delta)
else:
    raise AssertionError

args = EinsumNetwork.Args(
    num_var=height*width,
    num_dims=1,
    num_classes=1,
    num_sums=num_sums,
    num_input_distributions=num_sums,
    exponential_family=exponential_family,
    exponential_family_args=exponential_family_args,
    online_em_frequency=online_em_frequency,
    online_em_stepsize=online_em_stepsize)

result_path = os.path.join(result_path, "sums_{}__inputs_{}".format(num_sums, num_sums))

print()
print(result_path)

if training:
    utils.mkdir_p(result_path)
    lock_file = result_path + "/file.lock"
    done_file = result_path + "/file.done"
    lock = filelock.FileLock(lock_file)
    try:
        lock.acquire(timeout=0.1)
        if os.path.isfile(done_file):
            print('Model is trained')
            pass
        else:
            spn = EinsumNetwork.EinsumNetwork(rg, args)
            spn.initialize()
            spn.to(device)
            print('NUMBER OF PARAMETERS: ', get_n_params(spn))
            print(spn)

            try:
                ret = train(spn,
                            train_loader,
                            valid_loader,
                            test_loader,
                            result_path,
                            train_time_limit,
                            worker_time_limit - (time.time() - worker_start))

            except Exception as e:
                logging.error(traceback.format_exc())
                print('Failure')

            os.system("touch {}".format(done_file))
        lock.release()
    except filelock.Timeout:
        print('filelock timeout')

else:
    spn = EinsumNetwork.EinsumNetwork(rg, args)
    spn.initialize()
    spn.to(device)
    spn = load(spn, result_path)
