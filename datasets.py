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
    BATCH = getattr(args, "gen_batch_size", 10000)
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
    print("HERE some data",[el.val for el in data[:5]])
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

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(self.chars)} # bijection 'V13' <-> 13
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping: 13 -> 'V13'

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ','.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

# -----------------------------------------------------------------------------

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

