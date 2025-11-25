from concurrent.futures import ProcessPoolExecutor
from slurm import init_signal_handler, init_distributed_mode
from utils import bool_flag, initialize_exp
from logging import getLogger
import numpy as np
import time
import random
import statistics
import torch
from environment import do_score, do_stats
from itertools import repeat, chain

import torch
from typing import Any, List, Optional

from threerank import GroupClass
from Sidon  import SidonSetDataPoint
from triangle import TriangleDataPoint
from square import SquareDataPoint
from model import CharDataset, Transformer, InfiniteDataLoader, evaluate, generate
import os
import argparse
import pickle

logger = getLogger()


def get_parser():
    parser = argparse.ArgumentParser('A simple PatternBoost loop for different maths problems')
    
    parser.add_argument('--sample_only', type=int, default=500000, help="sample the specified number from the model in each loop")
    parser.add_argument('--gensize', type=int, default=100000, help='Number of generate initial values')
    parser.add_argument('--max_int', type=int, default=1000000000000, help='maximum integer')
    parser.add_argument('--pop_size', type=int, default=200000, help='New examples at each epoch')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--ntest', type=int, default=1000, help='Size of test set')
    parser.add_argument('--base', type=int, default=10, help='Encoding base')
    parser.add_argument('--reverse', type=bool_flag, default=False, help='Reversed digits')
    parser.add_argument('--max_len', type=int, default=500, help='Block size, maximumlength of sequences')
    parser.add_argument('--task', type=str, default="GroupClass", help='Math problem to be addressed')
    parser.add_argument('--input_file', type=str, default="", help='Optional input file with data')
    parser.add_argument('--process_pool', type=bool_flag, default="true", help='use process_pool to generate and score initial data')
    parser.add_argument('--always_search', type=bool_flag, default="true", help='if True, use local search for all examples generated (if False, only for invalid examples)')

    #Generation arguments

    ### ADD HERE THE PARAMETERS SPECIFIC TO YOUR PROBLEM ###

    parser.add_argument('--symbols', type=str, default="|,&", help="symbols specific to the problem, separated by ',' ")

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
    parser.add_argument('--sidon_steps', type=int, default=2000, help='number of steps in local search')

    #TriangleFree
    parser.add_argument('--triangle_N', type=int, default=30, help='Number of vertices in the triangle-free graph')
    parser.add_argument('--triangle_hard', type=bool_flag, default="false", help='whether only triang-free graphs are accepted')
    parser.add_argument('--triangle_init_method', type=str, default="edge_removal", help='method of generation')

    #SquareFree
    parser.add_argument('--square_N', type=int, default=30, help='Number of vertices in the square-free graph')
    parser.add_argument('--square_hard', type=bool_flag, default="false", help='whether only square-free graphs are accepted')
    parser.add_argument('--square_init_method', type=str, default="edge_removal", help='method of generation')

    # Makemore params
    parser.add_argument('--num_workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max_steps', type=int, default=50000, help="max number of optimization steps to run for, or -1 for infinite.")
    # parser.add_argument('--max_epochs', type=int, default= 30000, help='number of epochs')
    parser.add_argument('--seed', type=int, default=-1, help="seed")
    # sampling
    parser.add_argument('--top_k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--n_layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n_head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n_embd', type=int, default=128, help="number of feature channels in the model")
    # optimization
    parser.add_argument('--batch_size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning_rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument('--max_output_length', type=int, default=100, help="maximum output length")
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

train_classes = {"GroupClass":GroupClass, "Sidon":SidonSetDataPoint, "Triangle":TriangleDataPoint, "Square":SquareDataPoint}

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
    if args.process_pool:
        BATCH = getattr(args, "gen_batch_size", 10000)
        batch_counts = [BATCH] * (args.gensize // BATCH)
        rem = args.gensize % BATCH
        if rem:
            batch_counts.append(rem)
        with ProcessPoolExecutor(max_workers=min(20,args.num_workers)) as executor:
            # map returns lists; stream them to avoid a giant materialization
            for chunk in executor.map(_worker_batch,
                                repeat(args, len(batch_counts)),
                                repeat(classname, len(batch_counts)),
                                repeat(_generate_and_score, len(batch_counts)),
                                batch_counts):
                if chunk:  # extend incrementally to manage memory
                    data.extend(chunk)
    else:
        for _ in range(args.gensize):
            d = _generate_and_score(args,classname)
            if d is not None:
                data.append(d)
    print("HERE some data",[el.val for el in data[:5]])
    return data

def _generate_and_score(args,classname):
    d = classname(args.val,args)
    d.calc_features()
    d.calc_score()
    return d if d.score >=0 else None

def _worker_batch(args, classname, method, n):
    out = []
    for _ in range(n):
        d = method(args, classname)
        if d is not None:
            out.append(d)
    return out

def _detokenize(data, args, classname, base, reverse):
    """
    Worker function for detokenizing a batch of data
    """
    out = []
    for d in data:
        lst = d.split(',')
        l = decode(lst=lst, args=args, classname=classname, base=base, reverse=reverse)
        if l is not None:
            out.append(l)
    return out


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


def decode(lst, args, classname, base=10, reverse=False)-> Optional[Any]:
    """
    Decode a list of tokens to return a DataPoint classname with the corresponding discriminant. Note: only reads the determinant and do not return the ap
    """
    return classname(args.val,args).decode(lst,base=base,reverse=reverse)

def detokenize(data, args, classname, base, reverse):
    res = []
    if args.process_pool:
        BATCH = getattr(args, "gen_batch_size", 10000)
        batch_counts = [BATCH] * (len(data) // BATCH)
        rem = len(data) % BATCH
        if rem:
            batch_counts.append(rem)
        # Create data slices for each batch
        data_slices = []
        start = 0
        for batch_size in batch_counts:
            end = start + batch_size
            data_slices.append(data[start:end])
            start = end
        with ProcessPoolExecutor(max_workers=min(20, args.num_workers)) as executor:
            # map returns lists; stream them to avoid a giant materialization
            for chunk in executor.map(_detokenize,
                                data_slices,
                                repeat(args, len(batch_counts)),
                                repeat(classname, len(batch_counts)),
                                repeat(base, len(batch_counts)),
                                repeat(reverse, len(batch_counts)),
                                ):
                if chunk:  # extend incrementally to manage memory
                    res.extend(chunk)
    else:
        for _,d in enumerate(data):
            lst = d.split(',')
            l = decode(lst=lst, args=args, classname=classname, base=base, reverse=reverse)
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

        logits, loss, _ = model(X, Y)
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
            curr_loss = 0
#            print_samples(num=10)
                
        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break
    
    if args.device == "cuda":
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
    elif args.device == "mps":
        logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB")

def generate_sample(model, train_dataset):
    new_words = []

    sample_batch_size =args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
    todo = args.sample_only // sample_batch_size
    for i in range(todo):
        if i % 100 == 0 :
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

    if os.environ.get("MODAL_EXP_ID") is None:
        os.environ["MODAL_EXP_ID"] = time.strftime("%Y_%m_%d_%H_%M_%S")

    parser = get_parser()
    args = parser.parse_args()

    args.device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cuda")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "mps":
        torch.mps.manual_seed(args.seed)

    init_distributed_mode(args)
    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    if args.is_slurm_job:
        init_signal_handler()
    
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    classname = train_classes[args.task]
    if classname == TriangleDataPoint and args.base == -1:
        args.base = args.triangle_N * (args.triangle_N - 1) // 2
    if classname == SquareDataPoint and args.base == -1:
        args.base = args.square_N * (args.square_N - 1) // 2
    # system inits
    torch.manual_seed(args.seed)
    # os.makedirs(args.work_dir, exist_ok=True)
    symbols = [str(i) for i in range(max(args.base,3))]
    symbols.extend(args.symbols.split(","))
    args.vocab_size = len(symbols) + 1
    args.block_size = args.max_len

    #Initialize class to be adressed

    #Initialize transformer
    model = Transformer(args)
    model.to(args.device)
    model_path = os.path.join(args.dump_path, "model.pt")
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")
    if os.path.isfile(model_path): 
        logger.info("resuming from existing model")

        if args.device == "cuda":
            reloaded = torch.load(model_path)
        elif args.device == "mps":
            reloaded = torch.load(model_path, map_location=torch.device('mps'))
        else:
            reloaded = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(reloaded, dict) and "state_dict" in reloaded:
            model.load_state_dict(reloaded["state_dict"])
        else:
            model.load_state_dict(reloaded)
    if os.path.isfile(train_data_path):
        logger.info("resuming from existing data")
        train_set = pickle.load(open(train_data_path, "rb"))
        test_set = pickle.load(open(test_data_path, "rb"))
    else:
        data = generate_and_score(args,classname=classname)
        data = select_best(args.pop_size, data)
    
        #Create datasets
        train_set, test_set = make_train_test(data, args.ntest)
        logger.info(f"Initial train and test generated. Size are train: {len(train_set)}, test {len(test_set)}")
        pickle.dump(train_set, open(train_data_path, "wb"))
        pickle.dump(test_set, open(test_data_path, "wb"))

    #log initial stats
    do_stats(-1,data=train_set)

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

        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8, fused=True)
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

        train(model, batch_loader,optimizer, test_dataset)

        new_words = generate_sample(model, train_dataset)
        logger.info(f"New words generated length is {len(new_words)}")

        # decode 
        new_data = detokenize(data=new_words, args=args, classname=classname, base=args.base, reverse=args.reverse)
        logger.info(f"New data detokenized length is {len(new_data)}")

        new_data = do_score(new_data,process_pool=args.process_pool,num_workers=args.num_workers,always_search=args.always_search)

        #Possible to add another generation method here and mix it before taking the best

        new_data = select_best(args.pop_size, new_data)

        new_train, test_set = make_train_test(new_data, args.ntest)
        logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
        #Get all examples of previous train and current train and then select best.
        train_set = select_best(args.pop_size, train_set + new_train)
    
        pickle.dump(test_set, open(test_data_path, "wb"))
        pickle.dump(train_set, open(train_data_path, "wb"))
    
        n_epoch += 1





    