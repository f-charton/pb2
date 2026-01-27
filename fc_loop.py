from concurrent.futures import ProcessPoolExecutor

import psutil
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
import psutil
import gc
import ctypes
import threading
import queue

logger = getLogger()


# TODO: is this necessary?
def force_release_memory():
    # multiple gc.collect() in case of looped calls
    gc.collect()
    gc.collect()
    gc.collect()
    
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


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
    parser.add_argument('--redeem', type=bool_flag, default="true", help='if True, save invalid examples)')
    parser.add_argument('--mutation', type=int, default=0, help="max number of mutations to apply to the data")
    parser.add_argument('--offline_curriculum', type=bool_flag, default="false", help='if True, use data from N-1 to generate data for N. Applicable only if min_N < max_N')
    parser.add_argument('--n_unique_searches', type=int, default=1, help="how many distinct searches to perform from a given sample")
    
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
    parser.add_argument('--temp_span', type=int, default=0, help="temperature span")
    parser.add_argument('--inc_temp', type=float, default=0.0, help="temperature")
    parser.add_argument('--scoring_method', type=str, default="top", help="scoring method: you can choose between top, pow, exp")
    parser.add_argument('--scoring_pow_score', type=float, default=1.0, help="power score for scoring: if scoring_method is pow, this is the power")
    parser.add_argument('--scoring_exp_score', type=float, default=1.0, help="exp score for scoring: if scoring_method is exp, this is the multiplier for the exp")
    parser.add_argument('--keep_only_unique', type=bool_flag, default="true", help='keep only unique data')
    parser.add_argument("--always_reload", type=bool_flag, default="false",help="reload best model before generation")
    parser.add_argument("--save_best", type=bool_flag, default="true", help="save best model based on test loss")
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


def detokenize(data, args, env, executor=None):
    res = []
    pars = env.tokenizer.dataclass._save_class_params()
    if args.process_pool:
        BATCH = args.gen_batch_size
        data_slices = [data[i:i + BATCH] for i in range(0, len(data), BATCH)]
        
        if executor is not None:
            for chunk in executor.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                if chunk:
                    res.extend(chunk)
        else:
            with ProcessPoolExecutor(max_workers=min(20, args.num_workers)) as ex:
                # map returns lists; stream them to avoid a giant materialization
                for chunk in ex.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
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
            if args.save_best and test_loss < best_loss:
                model_path = os.path.join(args.dump_path, "model.pt")
                optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
                torch.save(model.state_dict(), model_path)
                torch.save(optim.state_dict(), optimizer_path)
                logger.info(f"test loss {test_loss} is the best so far, saved model to {model_path}")
                best_loss = test_loss
            curr_loss = 0

    if not args.save_best:
        model_path = os.path.join(args.dump_path, "model.pt")
        optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
        torch.save(model.state_dict(), model_path)
        torch.save(optim.state_dict(), optimizer_path)

    return best_loss
    

def sample_and_score(model, args, stoi, itos, env, temp, tempspan=0):
    sample_batch_size = args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
    todo = args.sample_only // sample_batch_size
    DETOK_CHUNK_SIZE = 10
    
    work_queue = queue.Queue()
    results_lock = threading.Lock()
    results = []
    
    total_invalid = 0
    all_processed_data = []
    
    executor = ProcessPoolExecutor(max_workers=min(20, args.num_workers))
    
    def consumer_thread():
        nonlocal total_invalid
        try:
            while True:
                batches = work_queue.get()
                
                if batches is None:
                    work_queue.task_done()
                    break
                
                all_data = [batch_numpy[j] for batch_numpy in batches for j in range(batch_numpy.shape[0])]
                del batches  # TODO: is this necessary?
                
                detok_results = detokenize(all_data, args, env, executor=executor)
                del all_data  # TODO: is this necessary?
                
                valid_data, n_invalid, processed_data = do_score(
                    detok_results, 
                    args=args,
                    executor=executor
                )
                
                del detok_results  # TODO: is this necessary?
                
                with results_lock:
                    results.extend(valid_data)
                    total_invalid += n_invalid
                    all_processed_data.extend(processed_data)

                del valid_data, processed_data  # TODO: is this necessary?
                
                work_queue.task_done()
                gc.collect()  # TODO: is this necessary?
                
        except Exception as e:
            logger.exception(f"Consumer thread error: {e}")
   
    consumer = threading.Thread(target=consumer_thread, daemon=True)
    consumer.start()
    
    pending_batches = []


    for i in range(todo):
        if tempspan > 0:
            curr_temp = temp+ 0.1*np.random.randint(tempspan+1)
        else:
             curr_temp = temp
        if i % 100 == 0:
            with results_lock:
                scored_so_far = len(results)
            logger.info(f'{i*sample_batch_size} / {todo * sample_batch_size} samples generated, {scored_so_far} scored')

        # X_init is a (batch, 1, token_embeddings) tensor where the first element is stoi[f"n{N}"] where N is a random number between min_N and max_N
        # sampling_Ns are batch_size numbers between min_N and max_N
        X_init = torch.empty((sample_batch_size, 1, args.token_embeddings), dtype=torch.long)
        for idx in range(sample_batch_size):
            # TODO: make sure we want to sample N uniformly
            N = np.random.randint(args.min_N, args.max_N + 1)
            X_init[idx, 0, :] = stoi[f"n{N}"]
        X_init = X_init.to(args.device)
        top_k = args.top_k if args.top_k != -1 else None
        batch_numpy = model.generate(X_init, args.max_len + 1, temperature=curr_temp, top_k=top_k, do_sample=True).cpu().numpy()
        del X_init  # TODO: is this necessary?
        
        pending_batches.append(batch_numpy)
        
        if len(pending_batches) >= DETOK_CHUNK_SIZE:
            work_queue.put(pending_batches)            
            pending_batches = []
            
    if pending_batches:
        work_queue.put(pending_batches)
    
    work_queue.put(None)
    consumer.join()
    
    executor.shutdown(wait=True)
    
    do_stats(total_invalid, all_processed_data)
    
    return results


def log_resources(label):
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent()
    logger.info(f"[{label}] CPU: {cpu_percent:.1f}% | RAM: {rss_mb:.1f}MB")


def write_important_metrics(metrics, epoch, metric_file, command=None, log_all_N=False):
    if metrics is not None:
        with open(metric_file, "a") as f:
            if command is not None:
                f.write(f"command: {command}\n")
            f.write(f"epoch: {epoch}\n")
            n_values = sorted(metrics.keys()) if log_all_N else [max(metrics.keys())]
            for n_value in n_values:
                f.write(f"N: {n_value} | mean: {metrics[n_value]['mean']} | median: {metrics[n_value]['median']} | top_1_percentile: {metrics[n_value]['top_1_percentile']} | max: {metrics[n_value]['max']}\n")
            f.write("--------------------------------\n")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.exp_id == "" and os.environ.get("MODAL_EXP_ID") is None:
        os.environ["MODAL_EXP_ID"] = time.strftime("%Y_%m_%d_%H_%M_%S")
        args.exp_id = os.environ["MODAL_EXP_ID"]

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
        logger.info("resuming from existing optimizer")
        if args.device == "cuda":
            reloaded = torch.load(optimizer_path)
        else:
            reloaded = torch.load(optimizer_path, map_location=torch.device(args.device))
        optimizer.load_state_dict(reloaded)
    
    train_set, test_set = load_initial_data(args, classname)
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")

    #log initial stats
    metrics = do_stats(-1,data=train_set)
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

    metric_file_interesting_N = os.path.join(args.dump_path, "metrics.txt")
    metric_file_all_N = os.path.join(args.dump_path, "metrics_all_N.txt")
    write_important_metrics(metrics, n_epoch, metric_file_interesting_N, command=args.command, log_all_N=False)
    write_important_metrics(metrics, n_epoch, metric_file_all_N, command=args.command, log_all_N=True)

    for epoch in range(n_epoch, args.max_epochs):
        logger.info(f"==== Starting Epoch {n_epoch} =====")
        log_resources(f"Epoch {epoch} START")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()
        
        # tokenize 
        train_words = [env.tokenizer.encode(d) for d in train_set]
        test_words = [env.tokenizer.encode(d) for d in test_set]
        # data loaders
        train_dataset = CharDataset(train_words, args.max_len, stoi, token_embeddings=args.token_embeddings)
        test_dataset = CharDataset(test_words, args.max_len, stoi, token_embeddings=args.token_embeddings)

        del train_words  # TODO: is this necessary?
        del test_words  # TODO: is this necessary?
        gc.collect()  # TODO: is this necessary?

        if args.device == "cuda":
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB")
        
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
        best_loss = train(model, args, batch_loader,optimizer, test_dataset, current_best_loss=best_loss)
        log_resources(f"Epoch {epoch} AFTER_TRAIN")
        force_release_memory()

        # taking the best model based on the test loss
        if args.always_reload:
            reload_model_optimizer(args, model, optimizer)
        logger.info(f"Sample with temperature {temperature} to {temperature+0.1*args.temp_span}")
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        new_data = sample_and_score(model, args, stoi, itos, env, temperature, args.temp_span)
        log_resources(f"Epoch {epoch} AFTER_SAMPLE")
        # do_stats(-1, data=new_data) # Moved inside update_datasets before updating but after deduplicating.

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        #Possible to add another generation method here and mix it before taking the best
        train_set, test_set, inc_temp = update_datasets(args, new_data, train_set, test_set, train_data_path, test_data_path)

        # #debug
        # logger.info(f"HERE do_stats after updating datasets")
        # metrics_temp = do_stats(-1,data=train_set)

        log_resources(f"Epoch {epoch} AFTER_UPDATE_DATASETS")

        del new_data  # TODO: is this necessary?
        gc.collect()  # TODO: is this necessary?
        gc.collect()  # TODO: is this necessary?
        force_release_memory()  # TODO: is this necessary?
                

        #Possible to add another generation method here and mix it before taking the best
        if inc_temp and args.inc_temp>0.0:
            temperature += args.inc_temp

        metrics = do_stats(-1, data=train_set)
    
        n_epoch += 1
        with open(epoch_file, "w") as f:
            f.write(str(n_epoch))
        with open(temp_file, "w") as f:
            f.write(str(temperature))
   
        write_important_metrics(metrics, n_epoch, metric_file_interesting_N, log_all_N=False)
        write_important_metrics(metrics, n_epoch, metric_file_all_N, log_all_N=True)
