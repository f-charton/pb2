from slurm import init_signal_handler, init_distributed_mode
from utils import bool_flag, initialize_exp
from logging import getLogger
import numpy as np
from numba import njit
import time
import random
import statistics
import torch
from threerank import get_three_rank,legendre_symbol

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from typing import List, Optional

from makemoretokens import ModelConfig, CharDataset, Transformer, Bigram, MLP, RNN, BoW, InfiniteDataLoader, evaluate, generate
import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NB_AP=20
primes = [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73]
assert len(primes) > NB_AP

logger = getLogger()


def get_parser():
    parser = argparse.ArgumentParser('Generate training sample of low braids via reservoir sampling')
    
    parser.add_argument('--sample-only', type=int, default=500000, help="sample the specified number from the model in each loop")
    

    parser.add_argument('--gensize', type=int, default=1000000, help='Number of generate initial values')
    parser.add_argument('--max_int', type=int, default=1000000000000, help='maximum integer')
    parser.add_argument('--pop_size', type=int, default=10000, help='New examples at each epoch')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--ntest', type=int, default=5000, help='Size of test set')
    parser.add_argument('--base', type=int, default=10, help='Encoding base')
    parser.add_argument('--reverse', type=bool_flag, default=False, help='Reversed digits')
    parser.add_argument('--nb_ap', type=int, default=10, help='Number of ap')
    parser.add_argument('--max_len', type=int, default=150, help='Block size, maximumlength of sequences')


    # Makemore params
    parser.add_argument('--num-workers', '-n', type=int, default=8, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=20000, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--max_epochs', type=int, default= 30000, help='number of epochs')
    parser.add_argument('--seed', type=int, default=-1, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=8, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument('--max-output-length', type=int, default=160, help="maximum output length")
    parser.add_argument('--gen_batch_size', type=int, default=1000, help="generation batch size")
    parser.add_argument('--n_tokens', type=int, default=100, help="nr tokens in tokenizer")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
    

    # path and ports
    parser.add_argument("--dump_path", type=str, default="checkpoint",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--cpu", type=bool_flag, default="false",
                        help="run on cpu only")
# debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    return parser

class datapoint():
    """
    Main object class representing a group class. Contains:
     val: (absolute value of the) discriminant
     ap: list of the NB_AP first legendre symbols
     score: 3-rank
    """
    def __init__(self,val):
        self.val=val
        self.ap = [-2]*NB_AP
        self.score = -1
    def calc_ap(self):
        for i in range(NB_AP): #TODO
            self.ap[i] = legendre_symbol(self.val,primes[i]) + 1
    def calc_score(self):
        self.score=get_three_rank(self.val)


def do_score(data):
    """
    Does the local search, computate the score, and features (if relevant)
    """
    for d in data:
        d.calc_ap()
        d.calc_score()

    # Compute and log statistics
    valid_data = [d for d in data if d.score >= 0]
    invalid_data = [d for d in data if d.score < 0]
    scores = [d.score for d in valid_data]
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    max_score = max(scores)
    logger.info(f"### Score distribution ###")
    logger.info(f"Invalid examples: {len(invalid_data)}")
    logger.info(f"Valid examples {len(valid_data)}")
    logger.info(f"Mean score: {mean}")
    logger.info(f"Median score: {median}")
    logger.info(f"stdev score: {stdev}")
    logger.info(f"Max score: {max_score}")


    return [d for d in data if d.score >= 0]


def load_data(infile):
    data = []
    with open(infile, "r") as file:
        for line in file:
            dat,sc = line.split('\t')
            d = datapoint(int(dat))
            d.score = int(sc)
            d.calc_ap()
            data.append(d)
    return data

def generate_and_score(args):
    """
    Generation method if no data
    """
    initvals = np.random.randint(1, args.max_int, size=args.gensize, dtype=np.int64)
    data = []
    for v in initvals:
        d = datapoint(v) 
        d.calc_ap()
        d.calc_score()
        if d.score >= 0:
            data.append(d)
    return data

def select_best(n, data):
    """
    Select the n-best data shuffled
    """
    if len(data) <= n:
        return data
    return random.shuffle(data.sort(key=lambda x: x.score, reverse=True)[:n])

def encode(d,base=10, reverse=False) -> list[str]:
    """
    Encode the data as a list of tokens containing the ap and the value of the discriminant
    """
    lst = []
    for s in d.ap:
        lst.append(str(s))
    v = d.val
    w = []
    while d >0:
        w.push_back(str(d%base))
        d=d//base
    if reverse:
        return lst + w
    else:
        return lst + w[::-1]

def decode(lst, base=10, reverse=False)-> Optional[datapoint]:
    """
    Decode a list of tokens to return a datapoint with the corresponding discriminant. Note: only reads the determinant and do not return the ap
    """
    if len(lst) <= NB_AP + 1:
        return None
    lst = lst[NB_AP:]
    val=0
    if reverse:
        try:
            for d in lst[::-1]:
                v = int(d)
                if v<0 or v>=base:
                    return None
                val = val*base + v
        except:
            return None
    else:
        try:
            for d in lst:
                v = int(d)
                if v<0 or v>=base:
                    return None
                val = val*base + v
        except:
            return None
    return datapoint(val)

def detokenize(data, base, reverse):
    res = []
    for d in data:
        l = decode(d,base,reverse)
        if l is None:
            continue
        res.append(l)
    return res

def make_train_test(data,ntest):
    """
    Create a train and test dataset from a dataset.
    """
    rp = np.random.permutation(data)
    return rp[:-ntest], rp[-ntest:]


def write_samples(num=10, new_file=False, use_logger=False):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, temperature = args.temperature, top_k=top_k, do_sample=True).to('cpu')
    #logger.info(f"generated")
    n_samp =0
    max_samp=0
    sum_samp=0
    samples = []
#    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        samples.append(word_samp)
    for s in samples:
        n_samp +=1
        sum_samp += len(s)
        max_samp = max(max_samp, len(s))
    out_file = args.dump_path + "/out.txt"
    #if use_logger:
        #logger.info("decoded")
        # logger.info(f"Printing {len(samples)} samples to {out_file}.")
    #else: 
        # print(f"Printing {len(samples)} samples to {out_file}.")
    if not new_file:
        with open(out_file, "a") as file:
            for word in samples:
                file.write(word)
                file.write("\n")
    else:
        with open(out_file, "w") as file:
            for word in samples:
                file.write(word)
                file.write("\n")
    #logger.info("printed")
    return n_samp, sum_samp, max_samp


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    init_distributed_mode(args)
    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    if args.is_slurm_job:
        init_signal_handler()
    
    args.device = "cpu" if args.cpu else "cuda"
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # os.makedirs(args.work_dir, exist_ok=True)
    symbols = [str(i) for i in range(max(args.base,3))]
    args.vocab_size = len(symbols) + 1
    args.block_size = args.max_len

    # Initialize the data
    if args.input_file != '':
        data = load_data(args.input_file)
    else: 
        data = generate_and_score(args)
    args.gen_size = len(data)
    
    data = select_best(args.pop_size, data)

    #Initialize transformer
    model = Transformer(args)
    model.to(args.device)
    model_path = os.path.join(args.dump_path, "model.pt")
    if os.path.isfile(model_path): # Note: if we sample-only then we also assume we are resuming
        logger.info("resuming from existing model")
        model.load_state_dict(torch.load(model_path))
    
    #Create datasets
    train, test = make_train_test(data, args.ntest)
    logger.info(f"Initial train and test generated. Size are train: {len(train)}, test {len(test)}")

    # Loop of PatternBoost
    n_epoch = 0
    for epoch in range(args.max_epochs):
        logger.info(f"==== Starting Epoch {n_epoch} =====")
        # tokenize 
        train_words = [encode(d,args.base,args.reverse) for d in train]
        test_words = [encode(d,args.base,args.reverse) for d in test]
        # data loaders
        train_dataset = CharDataset(train_words, symbols, args.max_output_length)
        test_dataset = CharDataset(test_words, symbols, args.max_output_length)


        model.train()
        new_words = model.generate()

        # decode 
        new_data = detokenize(new_words,args.base,args.reverse) 

        new_data = do_score(new_data)

        #Possible to add another generation method here and mix it before taking the best

        new_data = select_best(args.pop_size, new_data)

        new_train, test = make_train_test(new_data, args.ntest)
        logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test)}")
        #Get all examples of previous train and current train and then select best.
        train = select_best(args.pop_size, train + new_train)
        n_epoch += 1





    ##### TO BE UPDATED #####

    # init datasets
    for i in range(1,args.max_epochs):
        if not os.path.isfile(f"{args.dump_path}/search_output_{i}-tokenized.txt"):
            break
    initial_gen = i-1
    if initial_gen == 0:
        tokenize(f"{args.dump_path}/search_output_1.txt", args.n_tokens)
        initial_gen = 1
    
    logger.info(f"initializing at generation: {initial_gen}")
    input_file = args.dump_path + f"/search_output_{initial_gen}-tokenized.txt"
    train_dataset, test_dataset = create_datasets(input_file)
    vocab_size = args.n_tokens + 1
    block_size = args.max_output_length + 1
    logger.info(f"dataset determined that: {vocab_size=}, {block_size=}")

    init_model()
    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
    model = Transformer(config)
    model.to(args.device)
    logger.info(f"model #params: {sum(p.numel() for p in model.parameters())}")
    model_path = os.path.join(args.dump_path, "model.pt")
    if os.path.isfile(model_path): # Note: if we sample-only then we also assume we are resuming
        logger.info("resuming from existing model")
        model.load_state_dict(torch.load(model_path))


    for generation in range(initial_gen,args.max_epochs + 1):
        logger.info(f"============ Start of generation {generation} ============")
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")

        logger.info("training")
        # python makemoretokens.py --i search_output_1-tokenized.txt --device cuda
        #train_makemore()
        # init optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

        # init dataloader
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

        # training loop
        best_loss = None
        step = 0
        while True:

            t0 = time.time()

            # get the next batch, ship to device, and unpack it to input and target
            batch = batch_loader.next()
            batch = [t.to(args.device) for t in batch]
            X, Y = batch

            # feed into the model
            try:
                logits, loss = model(X, Y)
                # calculate the gradient, update the weights
                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            except RuntimeError as e:
                logger.info("Caught RuntimeError during forward pass.")
                logger.info(f"Shape of x before error: {X.shape}")
                logger.info(f"Shape of y before error: {Y.shape}")
                logger.info(f"Shape of logits (if calculated): {logits.shape if 'logits' in locals() else 'Not calculated'}")

                #raise e

            

            # wait for all CUDA work on the GPU to finish then calculate iteration time taken
            if args.device =="cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            # logging
            if step % 100 == 0:
                logger.info(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

            # evaluate the model
            if step > 0 and step % 500 == 0:
                train_loss = evaluate(model, train_dataset, args.device, batch_size=100, max_batches=10)
                test_loss  = evaluate(model, test_dataset,  args.device, batch_size=100, max_batches=10)
                logger.info(f"step {step} train loss: {train_loss} test loss: {test_loss}")
                # save the model to disk if it has improved
                if best_loss is None or test_loss < best_loss:
                    out_path = os.path.join(args.dump_path, "model.pt")
                    logger.info(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                    torch.save(model.state_dict(), out_path)
                    best_loss = test_loss
    #            print_samples(num=10)
                    
            step += 1
            # termination conditions
            if args.max_steps >= 0 and step >= args.max_steps:
                break
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")

        logger.info('generating')
        sample_batch_size =args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
        todo = args.sample_only
        tot_n = 0
        tot_sum = 0
        tot_max = 0
        out_file = args.dump_path + "/out.txt"
        in_file = args.dump_path + f"/search_output_{generation}-tokenized.txt"
        #infilz = f"{args.dump_path}/search_output_{generation}.txt"
        with open(in_file, 'r') as f:
            data = f.read()
        words = data.splitlines()
        with open(out_file, "w") as file:
            for word in words:
                file.write(word)
                file.write("\n")
        while sample_batch_size < todo:
            if todo % 50000 ==0 : 
                logger.info(f'{todo} samples remaining')
            n, sm, mx = write_samples(num=sample_batch_size)
            tot_n+=n
            tot_sum+=sm
            tot_max = max(tot_max,mx)
            todo = todo - sample_batch_size
        n, sm, mx = write_samples(num=todo)
        tot_n+=n
        tot_sum+=sm
        tot_max = max(tot_max,mx)
        logger.info(f"distribution of sample lengths: average: {tot_sum/tot_n if tot_n != 0 else 0} max: {tot_max}")
        logger.info('decoding')
        decode()
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        logger.info(f"============ End of generation {generation} ============")
        if os.path.exists(args.dump_path+"/distribution.txt"):
            with open(args.dump_path+"/distribution.txt", 'r') as file:
                d_lines = file.readlines()
        logger.info("distribution of scores")
        for l in d_lines:
            logger.info(l[:-1])

        
        logger.info("tokenizing")
        tokenize(f"{args.dump_path}/search_output_{generation+1}.txt", args.n_tokens)
        input_file = args.dump_path + f"/search_output_{generation+1}-tokenized.txt"
        train_dataset, test_dataset = create_datasets(input_file)
        

