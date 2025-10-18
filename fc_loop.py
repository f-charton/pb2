from slurm import init_signal_handler, init_distributed_mode
from utils import bool_flag, initialize_exp
from logging import getLogger
import numpy as np
import time
import random
import statistics
import torch
from environment import do_score

import torch
from typing import Any, List, Optional

from threerank import GroupClass
from Sidon  import SidonSetDataPoint
from model import CharDataset, Transformer, InfiniteDataLoader, evaluate, generate
import os
import argparse

logger = getLogger()


def get_parser():
    parser = argparse.ArgumentParser('A simple PatternBoost loop for different maths problems')
    
    parser.add_argument('--sample-only', type=int, default=500000, help="sample the specified number from the model in each loop")
    parser.add_argument('--gensize', type=int, default=1000000, help='Number of generate initial values')
    parser.add_argument('--max_int', type=int, default=1000000000000, help='maximum integer')
    parser.add_argument('--pop_size', type=int, default=10000, help='New examples at each epoch')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--ntest', type=int, default=5000, help='Size of test set')
    parser.add_argument('--base', type=int, default=10, help='Encoding base')
    parser.add_argument('--reverse', type=bool_flag, default=False, help='Reversed digits')
    parser.add_argument('--max_len', type=int, default=500, help='Block size, maximumlength of sequences')
    parser.add_argument('--task', type=str, default="GroupClass", help='Math problem to be addressed')
    parser.add_argument('--input_file', type=str, default="", help='Optional input file with data')

    #Generation arguments

    ### ADD HERE THE PARAMETERS SPECIFIC TO YOUR PROBLEM ###

    parser.add_argument('--symbols', type=str, default="|", help="symbols specific to the problem, separated by ',' ")

    #GroupClass
    parser.add_argument('--val', type=int, default=-1, help='absolute value of the discriminant, generated randomly if -1')
    parser.add_argument('--nb_ap', type=int, default=10, help='Number of ap')

    #SidonSets
    parser.add_argument('--N', type=int, default="500", help='Defines the set {0,....,N} in which the Sidon subset is looked for')
    parser.add_argument('--M', type=int, default="1", help='reward weight for length of Sidon Sets')
    parser.add_argument('--hard', type=bool_flag, default="true", help='whether only sidon sets are accepted')
    parser.add_argument('--insert_prob', type=float, default=0.33, help='probability of insert move in the local search')
    parser.add_argument('--delete_prob', type=float, default=0.33, help='probability of delete move in the local search')
    parser.add_argument('--shift_prob', type=float, default=0.33, help='probability of shift move in the local search')
    parser.add_argument('--temp0', type=float, default=0.33, help='temp0 of the local search')
    parser.add_argument('--temp_decay', type=float, default=0.33, help='temp_decay of the local search')
    parser.add_argument('--init_method', type=str, default="random_greedy", help='method of generation')
    parser.add_argument("--init_k", type=int, default=-1, help="by default size of the Sidon set that one tries to construct in the generation")
    parser.add_argument("--jitter_init", type=bool_flag, default="true", help="if generation is evenly spaced, should there be random displacements")
    parser.add_argument('--sidon_steps', type=int, default=200000, help='number of steps in local search')



    # Makemore params
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=20000, help="max number of optimization steps to run for, or -1 for infinite.")
    # parser.add_argument('--max_epochs', type=int, default= 30000, help='number of epochs')
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

train_classes = {"GroupClass":GroupClass, "Sidon":SidonSetDataPoint}

def load_data(infile, classname):
    data = []
    with open(infile, "r") as file:
        for line in file:
            d = classname.from_string(line)
            data.append(d)
    return data

def generate_and_score(args, classname):
    """
    Generation method if no data
    """
    data = []
    for v in range(args.gensize):
        d = classname(args.val,args)
        d.calc_features()
        d.calc_score()
        if d.score >= 0:
            data.append(d)
    print("HERE some data",[el.val for el in data[:5]])
    return data

def select_best(n, data):
    """
    Select the n-best data shuffled
    """
    to_shuff = data.copy()
    if len(data) <= n:
        return data
    to_shuff.sort(key=lambda x: x.score, reverse=True) # sort method returns None
    to_shuff = to_shuff[:n]
    random.shuffle(to_shuff)
    return to_shuff

def encode(d,base=10, reverse=False) -> list[str]:
    """
    Encode the data as a list of tokens containing the ap and the value of the discriminant
    """
    return d.encode(base=base,reverse=reverse)


def decode(lst, params, classname, base=10, reverse=False)-> Optional[Any]:
    """
    Decode a list of tokens to return a DataPoint classname with the corresponding discriminant. Note: only reads the determinant and do not return the ap
    """
    return classname(args.val,params).decode(lst,base=base,reverse=reverse)

def detokenize(data, params, classname, base, reverse):
    res = []
    for i,d in enumerate(data):
        lst = d.split(',')
        l = decode(lst=lst, params=params, classname=classname, base=base,reverse=reverse)
        if l is None:
            continue
        res.append(l)
    return res

def make_train_test(data,ntest):
    """
    Create a train and test dataset from a dataset.
    """
    indices = np.random.permutation(len(data))
    rp = [data[i] for i in indices]
    return rp[:-ntest], rp[-ntest:]

def train(model, loader, optim, test_dataset):
    # training loop
    best_loss = None
    step = 0
    curr_loss = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        curr_loss += loss.item()
        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device =="cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 100 == 0:
            logger.info(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = curr_loss / 500 
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

def generate_sample(model, train_dataset):
    new_words = []

    sample_batch_size =args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
    todo = args.sample_only // sample_batch_size
    for i in range(todo):
        if i % 100 ==0 : 
            logger.info(f'{i*sample_batch_size} / {todo * sample_batch_size} samples generated')
    
        X_init = torch.zeros(sample_batch_size, 1, dtype=torch.long).to(args.device)
        top_k = args.top_k if args.top_k != -1 else None
        X_samp = generate(model, X_init, args.max_output_length, temperature = args.temperature, top_k=top_k, do_sample=True).to('cpu')

        for i in range(X_samp.size(0)):
            # get the i'th row of sampled integers, as python list
            row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
            # token 0 is the <STOP> token, so we crop the output sequence at that point
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
            word_samp = train_dataset.decode(row)
            new_words.append(word_samp)
    return new_words


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
    symbols.extend(args.symbols.split(","))
    args.vocab_size = len(symbols) + 1
    args.block_size = args.max_len

    #Initialize class to be adressed
    classname = train_classes[args.task]

    #Initialize transformer
    model = Transformer(args)
    model.to(args.device)
    model_path = os.path.join(args.dump_path, "model.pt")
    if os.path.isfile(model_path): 
        logger.info("resuming from existing model")
        model.load_state_dict(torch.load(model_path))
        init_train_dataset = CharDataset(words = [],chars=symbols,max_word_length=args.max_output_length)
        new_words = generate_sample(model,init_train_dataset)
        # decode 
        data = detokenize(data=new_words, params=args, classname=classname, base=args.base,reverse=args.reverse)
        data = do_score(data)
    else:    
        # Initialize the data
        if args.input_file != "":
            data = load_data(args.input_file, classname=classname)
        else: 
            data = generate_and_score(args,classname=classname)
    args.gen_size = len(data)
    
    data = select_best(args.pop_size, data)

    #Create datasets
    train_set, test_set = make_train_test(data, args.ntest)
    logger.info(f"Initial train and test generated. Size are train: {len(train_set)}, test {len(test_set)}")

    # Loop of PatternBoost
    n_epoch = 0
    for epoch in range(args.max_epochs):
        logger.info(f"==== Starting Epoch {n_epoch} =====")
        # tokenize 
        train_words = [encode(d,args.base,args.reverse) for d in train_set]
        print("HERE some train words",train_words[:5])
        test_words = [encode(d,args.base,args.reverse) for d in test_set]
        # data loaders
        train_dataset = CharDataset(train_words, symbols, args.max_output_length)
        test_dataset = CharDataset(test_words, symbols, args.max_output_length)

        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

        train(model, batch_loader,optimizer, test_dataset)

        new_words = generate_sample(model, train_dataset)

        # decode 
        new_data = detokenize(data=new_words,params=args,classname=classname, base=args.base,reverse=args.reverse)

        new_data = do_score(new_data)

        #Possible to add another generation method here and mix it before taking the best

        new_data = select_best(args.pop_size, new_data)

        new_train, test_set = make_train_test(new_data, args.ntest)
        logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
        #Get all examples of previous train and current train and then select best.
        train_set = select_best(args.pop_size, train_set + new_train)
        n_epoch += 1





    