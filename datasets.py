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
from collections import defaultdict

from envs.environment import do_stats


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
            for chunk in executor.map(classname._batch_generate_and_score, batch_counts, 
                                    repeat(args.min_N, len(batch_counts)), repeat(args.max_N, len(batch_counts)), repeat(pars, len(batch_counts))):
                if chunk:  # extend incrementally to manage memory
                    data.extend(chunk)
    else:
        for t in batch_counts:
            d = classname._batch_generate_and_score(t, args.min_N, args.max_N)
            if d is not None:
                data.extend(d)
    return data

def generate_from_existing_data(args, classname, data):
    res = []
    pars = classname._save_class_params()
    if args.process_pool:
        BATCH = args.gen_batch_size
        data_slices = [data[i:i + BATCH] for i in range(0, len(data), BATCH)]
        with ProcessPoolExecutor(max_workers=min(20, args.num_workers)) as ex:
            # map returns lists; stream them to avoid a giant materialization
            for chunk in ex.map(classname._batch_generate_from_existing_data, data_slices, repeat(args.mutation, len(data_slices)), repeat(pars, len(data_slices))):
                if chunk:  # extend incrementally to manage memory
                    res.extend(chunk)
    else:
        res = classname._batch_generate_from_existing_data(data, args.mutation, pars)
    return data


def _helper_select_best(n, data, scoring_method="top", pow_score=1.0, exp_score=1.0):
    if len(data) <= n:
        return data
    if scoring_method == "top":
        sorted_data = sorted(data, key=lambda x: x.score, reverse=True)
        return sorted_data[:n]

    scores = np.array([x.score for x in data])
    
    if scoring_method == "pow":
        weights = np.power(np.maximum(scores, 0), pow_score)
    elif scoring_method == "exp":
        max_score = scores.max()
        weights = np.exp(exp_score * (scores - max_score))
            
    probs = weights / weights.sum()
    indices = np.random.choice(len(data), size=n, replace=False, p=probs)
    return [data[i] for i in indices]

def select_best(n, data, scoring_method="top", pow_score=1.0, exp_score=1.0):
    """
    With scoring_method="top", we select the top n data
    With scoring_method="pow", sample with probability ∝ score^pow_score
    With scoring_method="exp", sample with probability ∝ exp(exp_score * (score - max_score))
    Maintains the proportion of N values from the original data.
    """
    if len(data) <= n:
        return data

    # This part is to ensure that we keep the proportion of Ns the same as the original data
    groups = defaultdict(list)
    for item in data:
        groups[item.N].append(item)
    total = len(data)
    n_values = list(groups.keys())
    exact_allocations = {nv: n * len(groups[nv]) / total for nv in n_values}
    allocations = {nv: int(exact_allocations[nv]) for nv in n_values}

    remaining = n - sum(allocations.values())
    fractional_parts = sorted(n_values, key=lambda nv: exact_allocations[nv] - allocations[nv], reverse=True)
    for i in range(remaining):
        allocations[fractional_parts[i]] += 1

    selected = []
    for nv, group in groups.items():
        selected.extend(_helper_select_best(allocations[nv], group, scoring_method, pow_score, exp_score))
    random.shuffle(selected)
    return selected


def make_train_test(data,ntest):
    """
    Create a train and test dataset from a dataset.
    """
    indices = np.random.permutation(len(data))
    rp = [data[i] for i in indices]
    return rp[:-ntest], rp[-ntest:]

def compute_unique_data(old_data, new_data=None):
    def add_unique(src, unique_hashes):
        des = []
        for d in src:
            if d.features not in unique_hashes:
                unique_hashes.add(d.features)
                des.append(d)
        return des, unique_hashes
    
    unique_hashes = set()
    unique_old_data, unique_hashes = add_unique(old_data, unique_hashes)
    if new_data is not None:
        unique_new_data, unique_hashes = add_unique(new_data, unique_hashes)
    else:
        unique_new_data = None
    return unique_old_data, unique_new_data


def update_datasets(args, data, train_set, test_set, train_path, test_path):
    inc_temp=False
    if args.keep_only_unique:
        bef = len(data)
        data, _ = compute_unique_data(data)
        aft = len(data)
        logger.info(f"Unique processing: {aft} examples left, {bef-aft} duplicates")
        do_stats(-1,data)
        if aft / (bef+1) < 0.9:
            inc_temp = True
    if args.new_proportion > 0.0:
        new_data = select_best(int(args.new_proportion*args.pop_size), data, args.scoring_method, args.scoring_pow_score, args.scoring_exp_score)
    else:
        new_data = select_best(args.pop_size, data, args.scoring_method, args.scoring_pow_score, args.scoring_exp_score)
    
    if len(new_data) >= 2* args.ntest or test_set is None:
        new_train, test_set = make_train_test(new_data, args.ntest)
    else:
        new_train = new_data
    logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
    #Get all examples of previous train and current train and then select best.
    if args.keep_only_unique:
        train_set, new_train = compute_unique_data(train_set, new_train)
        logger.info(f"Unique data computed for original train set: {len(train_set)}, generated train set: {len(new_train)}")
    if args.new_proportion > 0.0:
        train_set = select_best(int((1.0-args.new_proportion)*args.pop_size), train_set, args.scoring_method, args.scoring_pow_score, args.scoring_exp_score) + new_train
    else:    
        train_set = select_best(args.pop_size, train_set + new_train, args.scoring_method, args.scoring_pow_score, args.scoring_exp_score)
    logger.info(f"Final train and test generated. Size are train: {len(train_set)}, test {len(test_set)}")

    pickle.dump(test_set, open(test_path, "wb"))
    pickle.dump(train_set, open(train_path, "wb"))
    return train_set, test_set, inc_temp


def load_initial_data(args, classname):
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")
    if os.path.isfile(train_data_path):
        train_set = pickle.load(open(train_data_path, "rb"))
        test_set = pickle.load(open(test_data_path, "rb"))
        if args.offline_curriculum:
            data = train_set + test_set
            data = generate_from_existing_data(args, classname, data)
        else:
            logger.info("resuming from existing data")
    else:
        data = generate_and_score(args,classname=classname)
        test_set = []

        train_set = []
        train_set, test_set, _ = update_datasets(args, data, train_set, test_set, train_data_path, test_data_path)
    return train_set, test_set


class CharDataset(Dataset):
    def __init__(self, encoded_data, max_len, stoi, token_embeddings):
        self.encoded_data = encoded_data
        self.max_len = max_len
        self.eos_token_id = stoi["EOS"]
        self.pad_token_id = stoi["PAD"]
        self.token_embeddings = token_embeddings

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def collate_fn(self, batch):
        x = np.full((len(batch), self.max_len + 2, self.token_embeddings), self.pad_token_id, dtype=np.int32)

        for i, el in enumerate(batch):
            x[i, :el.shape[0], :] = el
            x[i, el.shape[0], :] = self.eos_token_id
        valid_col = (x != self.pad_token_id).any(axis=(0, 2))
        last_col = np.nonzero(valid_col)[0][-1] + 1
        x = x[:, :last_col, :]
        y = np.concatenate([x[:, 1:, :], np.full((len(batch), 1, self.token_embeddings), self.pad_token_id, dtype=x.dtype)], axis=1)
        return torch.LongTensor(x), torch.LongTensor(y)

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

