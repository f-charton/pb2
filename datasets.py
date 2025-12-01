import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import pickle
from logging import getLogger
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat, chain


logger = getLogger()

# helper functions for creating the training and test Datasets

def generate_and_score(args, classname):
    """
    Generation method if no data
    """
    data = []
    BATCH = args.gen_batch_size
    batch_counts = [BATCH] * (args.gensize // BATCH)
    rem = args.gensize % BATCH
    if rem:
        batch_counts.append(rem)
    if args.process_pool:
        pars = classname._save_class_params()
        with ProcessPoolExecutor(max_workers=min(20,args.num_workers)) as executor:
            # map returns lists; stream them to avoid a giant materialization
            for chunk in executor.map(classname._batch_generate_and_score,
                                batch_counts, repeat(pars, len(batch_counts))):
                if chunk:  # extend incrementally to manage memory
                    data.extend(chunk)
    else:
        for t in batch_counts:
            d = classname._batch_generate_and_score(t)
            if d is not None:
                data.extend(d)
    return data

def select_best(n, data):
    """
    Select the n-best data shuffled
    """
    if len(data) <= n:
        return data
    to_shuff = data.copy()
    to_shuff.sort(key=lambda x: x.score, reverse=True) # sort method returns None
    to_shuff = to_shuff[:n]
    random.shuffle(to_shuff)
    return to_shuff

def make_train_test(data,ntest):
    """
    Create a train and test dataset from a dataset.
    """
    indices = np.random.permutation(len(data))
    rp = [data[i] for i in indices]
    return rp[:-ntest], rp[-ntest:]



def update_datasets(args, data, train_set, train_path, test_path):
    new_data = select_best(args.pop_size, data)

    new_train, test_set = make_train_test(new_data, args.ntest)
    logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
    #Get all examples of previous train and current train and then select best.
    train_set = select_best(args.pop_size, train_set + new_train)

    pickle.dump(test_set, open(test_path, "wb"))
    pickle.dump(train_set, open(train_path, "wb"))
    return train_set, test_set


def load_initial_data(args, classname):
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")
    if os.path.isfile(train_data_path):
        logger.info("resuming from existing data")
        train_set = pickle.load(open(train_data_path, "rb"))
        test_set = pickle.load(open(test_data_path, "rb"))
    else:
        data = generate_and_score(args,classname=classname)

        train_set = []
        train_set, test_set = update_datasets(args, data, train_set,train_data_path, test_data_path)
    return train_set, test_set


class CharDataset(Dataset):
    def __init__(self, encoded_data, chars, max_len, stoi):
        self.encoded_data = encoded_data
        self.chars = chars
        self.max_len = max_len
        self.bos_token_id = stoi["BOS"]
        self.eos_token_id = stoi["EOS"]
        self.pad_token_id = stoi["PAD"]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def collate_fn(self, batch):
        x = torch.full((len(batch), self.max_len + 2), self.pad_token_id, dtype=torch.long)
        y = torch.full((len(batch), self.max_len + 2), self.pad_token_id, dtype=torch.long)
        x[:, 0] = self.bos_token_id
        for i, ix in enumerate(batch):
            x[i, 1 : len(ix) + 1] = ix
            y[i, : len(ix)] = ix
            x[i, len(ix) + 1] = self.eos_token_id
            y[i, len(ix)] = self.eos_token_id
        valid_col = (x != self.pad_token_id).any(dim=0)
        last_col = valid_col.nonzero(as_tuple=False)[-1].item() + 1
        x = x[:, :last_col]
        y = y[:, :last_col]
        return x, y

# -----------------------------------------------------------------------------

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, collate_fn=dataset.collate_fn, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

