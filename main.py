import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os 
import argparse
import torch
import numpy as np
from tqdm import tqdm 

from utils import preset, load_pretrained_model, load_noise_scheduler
from dataset import load_dataset
from jys import golden_section_algo, klub_algorithm1_t_q_data, klub_algorithm1_k_q_data
# from jys import golden_section_algo, klub_algorithm1_t_q_path, klub_algorithm1_k_q_path

def round_list(list_):
    return [round(i, 4) for i in list_]

if __name__ == "__main__":
    # 1. config
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--dataset_name", type=str, choices=["text", "webtext", "count", "cifar", "imagenet256", "imagenet512", "piano"])
    parser.add_argument("--model_name", type=str, choices=["d3pm", "sedd", "maskgit", "ctmc", "dfm"])
    parser.add_argument("--scheduler_name", type=str, default="euler", choices=["euler", "tweedie", "gillespie"])
    parser.add_argument("--noise_schedule", type=str, default="loglinear", choices=["loglinear", "cosine", "arccos"])
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--src_num_function_eval", type=int, default=-1)
    parser.add_argument("--tgt_num_function_eval", type=int, default=-1)
    parser.add_argument("--fix_length", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_iter", type=int, default=32)
    parser.add_argument("--gibbs_iter", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_path", type=str, default='../datasets')
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir = f"runs/{args.scheduler_name}/{args.dataset_name}-{args.model_name}"
    args.exp_name = f"sampling_schedule_list-nfe_{args.src_num_function_eval}-samples_{args.num_samples}"
    args.exp_name += f"-fix_length_{args.fix_length}" if args.fix_length != 0 else ""
    args.exp_name += f"-max_length_{args.max_length}" if args.max_length is not None else ""
    args = preset(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # NOTE: We need to use full-precision for KLUB optimization (half leads to numerical instability)
    dtype = torch.float 

    # 2. load pretrained model 
    model, args = load_pretrained_model(args)
    model.to(args.device, dtype)

    # 3. load dataset
    ds = load_dataset(args, model)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 4. load scheduler
    scheduler = load_noise_scheduler(args, model)

    # sample klub matrix
    sample_eps = 1e-5
    if args.scheduler_name == "gillespie":
        assert args.length == args.src_num_function_eval
        klub_vars = torch.linspace(0, args.length, 2).long().tolist()
        klub_vars_space = torch.linspace(0, args.length, args.src_num_function_eval + 1).long()
        klub_algorithm1_fn = klub_algorithm1_k_q_data 
        err = 0.25
    else:
        klub_vars = torch.linspace(1, sample_eps, 2).tolist()
        klub_vars_space = torch.linspace(1, sample_eps, args.src_num_function_eval + 1)
        klub_algorithm1_fn = klub_algorithm1_t_q_data 
        err = 0.25 / args.src_num_function_eval
    print('start optimizing the sampling schedule:', round_list(klub_vars))

    # load sampling schedule if exists
    if os.path.exists(f'{args.output_dir}/{args.exp_name}.pt'):
        sampling_schedule_list = torch.load(f'{args.output_dir}/{args.exp_name}.pt')
    else:
        sampling_schedule_list = [[0, args.src_num_function_eval]]
    
    # optimize sampling schedule
    for log_nfe in range(int(np.log2(args.tgt_num_function_eval).item())+1):
        # skip if sampling schedule already exists
        if 2**log_nfe+1 <= max([len(sampling_schedule) for sampling_schedule in sampling_schedule_list]):
            sampling_schedule = list(filter(lambda sampling_schedule: len(sampling_schedule)==2**log_nfe+1, sampling_schedule_list))[0]
            klub_vars = torch.tensor([klub_vars_space[min(i, args.src_num_function_eval)] for i in sampling_schedule]).tolist()
            print('*** skip nfe: {} ***'.format(2**log_nfe))
            continue
        else:
            print('*** current nfe: {} ***'.format(2**log_nfe))
            print('*** current sampling schedule: {} ***'.format(round_list(klub_vars)))

        # manual correction (avoid sampling schedule where consecutive steps have neighboring t)
        for i in range(len(sampling_schedule)-1):
            if sampling_schedule[i+1] - sampling_schedule[i] < 2:
                sampling_schedule[i+1] = min(sampling_schedule[i] + 2, args.src_num_function_eval)
        for i in range(len(sampling_schedule)-1,0,-1):
            if sampling_schedule[i] - sampling_schedule[i-1] < 2:
                sampling_schedule[i-1] = max(sampling_schedule[i] - 2, 0)
        klub_vars = torch.tensor([klub_vars_space[i] for i in sampling_schedule]).tolist()

        # optimization step
        v2_list = []
        for v1, v3 in tqdm(zip(klub_vars[:-1], klub_vars[1:]), desc='optimizing sampling schedule...'):
            f = lambda v2: -klub_algorithm1_fn(v1, v2, v3, model, scheduler, dl, 42, args.fix_length, args.graph, args.max_length, args.device, dtype)
            if args.scheduler_name == "gillespie":
                v2, v2_traj, klub_dict = golden_section_algo(f, v1+1+1e-3, v3-1-1e-3, err=err, max_iter=args.max_iter)
            if args.scheduler_name == "euler":
                v2, v2_traj, klub_dict = golden_section_algo(f, v3+args.eps, v1-args.eps, err=err, max_iter=args.max_iter)
            klub_traj = [klub_dict[v2].item() for v2 in v2_traj]
            v2_list.append(min(klub_dict, key=klub_dict.get))
        
        # update sampling schedule
        new_klub_vars = []
        for v, v_new in zip(klub_vars[:-1], v2_list):
            new_klub_vars.append(v)
            new_klub_vars.append(v_new)
        new_klub_vars += [klub_vars[-1]]

        # manaul correction (avoid the sampling schedule where different sampling step have same t)
        sampling_schedule = [(klub_vars_space - v2).abs().argmin().item() for v2 in new_klub_vars]
        for i in range(len(sampling_schedule)-1):
            if sampling_schedule[i+1] - sampling_schedule[i] < 1:
                sampling_schedule[i+1] = min(sampling_schedule[i]+1, args.src_num_function_eval+1)
        klub_vars = torch.tensor([klub_vars_space[i] for i in sampling_schedule]).tolist()

        # save optimized sampling schedule        
        sampling_schedule_list.append(sampling_schedule)
        torch.save(sampling_schedule_list, f'{args.output_dir}/{args.exp_name}.pt')