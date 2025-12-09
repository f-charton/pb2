from concurrent.futures import ProcessPoolExecutor
from slurm import init_signal_handler, init_distributed_mode
from utils import bool_flag, initialize_exp
from logging import getLogger
import numpy as np
# import multiprocessing as mp
import time
import torch
from envs.environment import do_score, do_stats
from itertools import repeat
from envs import ENVS, build_env

from models.model import  Transformer, evaluate
from datasets import CharDataset, InfiniteDataLoader, load_initial_data, update_datasets
import os
import argparse

logger = getLogger()


def get_parser():
    parser = argparse.ArgumentParser('A simple PatternBoost loop for different maths problems')
    
    parser.add_argument("--gensize", type=int, default=100000, help="Number of generate initial values")
    parser.add_argument("--sample_only", type=int, default=500000, help="sample the specified number from the model in each loop")
    parser.add_argument("--pop_size", type=int, default=200000, help="New examples at each epoch")
    parser.add_argument('--max_epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--ntest', type=int, default=1000, help='Size of test set')
    parser.add_argument('--max_len', type=int, default=500, help='Block size, maximum length of sequences')
    parser.add_argument('--env_name', type=str, default="sidon", help='Math problem to be addressed')
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    parser.add_argument('--input_file', type=str, default="", help='Optional input file with data')
    parser.add_argument('--process_pool', type=bool_flag, default="true", help='use process_pool to generate and score initial data')
    parser.add_argument('--always_search', type=bool_flag, default="true", help='if True, use local search for all examples generated (if False, only for invalid examples)')
    parser.add_argument('--new_proportion', type=float, default=0.0, help="proportion of new samples in test set")

    # Makemore params
    parser.add_argument('--num_workers', type=int, default=8, help="number of data workers for both train/test")
    parser.add_argument('--max_steps', type=int, default=50000, help="number of training steps.")
    parser.add_argument('--num_eval_steps', type=int, default=500, help="number of step between each evaluation during training.")
    parser.add_argument('--seed', type=int, default=-1, help="seed")
    # sampling
    parser.add_argument('--top_k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--n_layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n_head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n_embd', type=int, default=128, help="number of feature channels in the model")
    parser.add_argument('--no_positional', type=bool_flag, default="false", help='no positional embedding')
    
    # optimization
    parser.add_argument('--batch_size', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument('--gen_batch_size', type=int, default=1000, help="generation batch size")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
    parser.add_argument('--inc_temp', type=float, default=0.0, help="temperature")
    parser.add_argument('--keep_only_unique', type=bool_flag, default="true", help='keep only unique data')
    parser.add_argument("--always_reload", type=bool_flag, default="false",help="reload best model before generation")
# de

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


def detokenize(data, args, env):
    res = []
    pars = env.tokenizer.dataclass._save_class_params()
    if args.process_pool:
        BATCH = args.gen_batch_size
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
        # with ProcessPoolExecutor(max_workers=min(20, args.num_workers), mp_context=mp.get_context('spawn')) as executor:
        with ProcessPoolExecutor(max_workers=min(20, args.num_workers)) as executor:
            # map returns lists; stream them to avoid a giant materialization
            for chunk in executor.map(env.tokenizer.decode_batch, data_slices,repeat(pars, len(data_slices))):
                if chunk:  # extend incrementally to manage memory
                    res.extend(chunk)
    else:
        res = env.tokenizer.decode_batch(data, pars)
    return res


def reload_model_optimizer(args, model, optimizer):
    model_path = os.path.join(args.dump_path, "model.pt")
    optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
    if os.path.isfile(model_path):
        if args.device == "cuda":
            reloaded = torch.load(model_path)
        else:
            reloaded = torch.load(model_path, map_location=torch.device(args.device))
        model.load_state_dict(reloaded)
    if os.path.isfile(optimizer_path):
        if args.device == "cuda":
            reloaded = torch.load(optimizer_path)
        else:
            reloaded = torch.load(optimizer_path, map_location=torch.device(args.device))
        optimizer.load_state_dict(reloaded)
    logger.info("model and optimizer reloaded")


def train(model, args, loader, optim, test_dataset, current_best_loss=None):
    # training loop
    best_loss = current_best_loss or float("inf")
    step = 0
    curr_loss = 0
    for step in range(args.max_steps):

        if step % 100 == 0:
            t0 = time.time()
        batch = loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        _, loss, _ = model(X, Y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        curr_loss += loss.item()

        # logging
        if (step + 1) % 100 == 0:
            t1 = time.time()
            logger.info(f"step {step + 1} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")
        if (step + 1) % args.num_eval_steps == 0:
            train_loss = curr_loss / args.num_eval_steps
            test_loss = evaluate(model, test_dataset, args.device, batch_size=100, max_batches=10)
            logger.info(f"step {step + 1} train loss: {train_loss} test loss: {test_loss}")
            if test_loss < best_loss:
                model_path = os.path.join(args.dump_path, "model.pt")
                optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
                torch.save(model.state_dict(), model_path)
                torch.save(optim.state_dict(), optimizer_path)
                logger.info(f"test loss {test_loss} is the best so far, saved model to {model_path}")
                best_loss = test_loss
            curr_loss = 0

    return best_loss
    

def sample(model, args, stoi, itos, env, temp):
    eos_token_id = stoi["EOS"]
    bos_token_id = stoi["BOS"]

    new_words = []

    sample_batch_size = args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
    todo = args.sample_only // sample_batch_size
    for i in range(todo):
        if i % 100 == 0 :
            logger.info(f'{i*sample_batch_size} / {todo * sample_batch_size} samples generated')
    
        X_init = torch.full((sample_batch_size, 1, args.token_embeddings), bos_token_id, dtype=torch.long).to(args.device)
        top_k = args.top_k if args.top_k != -1 else None
        X_samp = model.generate(X_init, args.max_len + 1, temperature = temp, top_k=top_k, do_sample=True).to('cpu')
        
        for j in range(X_samp.size(0)):
            row = X_samp[j, 1:, :].tolist() # remove BOS token
            new_words.append(row)

    return detokenize(new_words, args, env)


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

    fused = True if args.device in ["cuda", "mps"] else False

    init_distributed_mode(args)
    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    if args.is_slurm_job:
        init_signal_handler()
    
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    env = build_env(args)

    classname = env.data_class
    
    # system inits
    torch.manual_seed(args.seed)

    args.vocab_size = len(env.tokenizer.itos)

    args.block_size = args.max_len + 2
    args.token_embeddings = env.tokenizer.token_embeddings
    stoi = env.tokenizer.stoi
    itos = env.tokenizer.itos

    #Initialize transformer
    model = Transformer(args, stoi["PAD"],stoi["EOS"], token_embeddings=args.token_embeddings)
    model.to(args.device)
    model_path = os.path.join(args.dump_path, "model.pt")
    optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
    if os.path.isfile(model_path): 
        logger.info("resuming from existing model")
        if args.device == "cuda":
            reloaded = torch.load(model_path)
        else:
            reloaded = torch.load(model_path, map_location=torch.device(args.device))
        model.load_state_dict(reloaded)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8, fused=fused)
    if os.path.isfile(optimizer_path):
        print("resuming from existing optimizer")
        if args.device == "cuda":
            reloaded = torch.load(optimizer_path)
        else:
            reloaded = torch.load(optimizer_path, map_location=torch.device(args.device))
        optimizer.load_state_dict(reloaded)
    
    train_set, test_set = load_initial_data(args, classname)
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")

    #log initial stats
    do_stats(-1,data=train_set)
    temperature = args.temperature
    # Loop of PatternBoost
    best_loss = None
    epoch_file = os.path.join(args.dump_path, "epoch.txt")
    if os.path.isfile(epoch_file):
        with open(epoch_file, "r") as f:
            n_epoch = int(f.read())
    else:
        n_epoch = 0
    temp_file = os.path.join(args.dump_path, "temperature.txt")
    if os.path.isfile(temp_file):
        with open(temp_file, "r") as f:
            temperature = float(f.read())
    else:
        temperature = args.temperature
    for epoch in range(n_epoch, args.max_epochs):
        logger.info(f"==== Starting Epoch {n_epoch} =====")
        # tokenize 
        train_words = [env.tokenizer.encode(d) for d in train_set]
        test_words = [env.tokenizer.encode(d) for d in test_set]
        # data loaders
        train_dataset = CharDataset(train_words, args.max_len, stoi, token_embeddings=args.token_embeddings)
        test_dataset = CharDataset(test_words, args.max_len, stoi, token_embeddings=args.token_embeddings)

        if args.device == "cuda":
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB")
        
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
        best_loss = train(model, args, batch_loader,optimizer, test_dataset, current_best_loss=best_loss)

        # taking the best model based on the test loss
        if args.always_reload:
            reload_model_optimizer(args, model, optimizer)
        logger.info(f"Sample with temperature {temperature}")
        new_data = sample(model, args, stoi, itos, env, temperature) # should the token decoder be in the dataset?
        logger.info(f"New data detokenized length is {len(new_data)}")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        new_data = do_score(new_data,process_pool=args.process_pool,num_workers=args.num_workers,always_search=args.always_search)

        #Possible to add another generation method here and mix it before taking the best
        train_set, test_set, inc_temp = update_datasets(args, new_data, train_set, train_data_path, test_data_path)
        if inc_temp and args.inc_temp>0.0:
            temperature += args.inc_temp

        do_stats(-1, data=train_set)
    
        n_epoch += 1
        with open(epoch_file, "w") as f:
            f.write(str(n_epoch))
        with open(temp_file, "w") as f:
            f.write(str(temperature))
   
