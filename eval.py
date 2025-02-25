# 1. load pretrained model (text generation: SEDD, image generation: CTMC)
# 2. eval (text generation: generative perplexity, image generation: FID)

import os 
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm 
from utils import flush, load_pretrained_model, preset, load_noise_scheduler, load_sampling_fn
from dataset import load_dataset, load_evaluation_fn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    # 1. config
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--dataset_name", type=str, choices=["text", "webtext", "count", "piano", "cifar"])
    parser.add_argument("--model_name", type=str, choices=["d3pm", "sedd", "maskgit", "ctmc"])
    parser.add_argument("--eval_dataset_name", type=str, default="")
    parser.add_argument("--scheduler_name", type=str, default="euler", choices=["euler", "tweedie", "gillespie", "pc"])
    parser.add_argument("--sampling_schedule_name", type=str, default="uniform", choices=["uniform", "jys"])
    parser.add_argument("--sampling_schedule_path", type=str, default="")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--fix_length", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ppl_batch_size", type=int, default=8)
    parser.add_argument("--src_nfe", type=int, default=512)
    parser.add_argument("--tgt_nfe", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--cond_num_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default='cuda:3')
    parser.add_argument("--save_dir", type=str, default='../runs-gen_x0', help="directory to save the generated samples")
    parser.add_argument("--output_dir", type=str, default='../runs-eval', help="directory to save the evaluation results")
    parser.add_argument("--dataset_path", type=str, default='../datasets')
    parser.add_argument("--cache_dir", type=str, default='../.cache')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args = preset(args)
    args.output_dir = f'{args.output_dir}/{args.dataset_name}-{args.model_name}-{args.scheduler_name}'
    args.save_dir = f'{args.save_dir}/{args.dataset_name}-{args.model_name}-{args.scheduler_name}'
    args.exp_name = f"{args.sampling_schedule_name}-tgt_nfe_{args.tgt_nfe}-src_nfe_{args.src_nfe}-seed_{args.seed}"
    args.exp_name += f"-fix_length_{args.fix_length}" if args.fix_length != 0 else ""
    args.exp_name += f"-max_length_{args.max_length}" if args.max_length is not None else ""
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # use full precision for evaluation
    dtype = torch.float32

    # 1. set seed
    seed_everything(args.seed)

    # 2. load pretrained model 
    model, args = load_pretrained_model(args)
    if args.model_name != 'maskgit':
        model.to(args.device, dtype)
        num_parameters = sum(p.numel() for p in model.parameters())

    # 3. load scheduler
    scheduler = load_noise_scheduler(args, model)
    ddms_sampling_fn = load_sampling_fn(args)

    # (optional) load eval dataset (in-context conditional generation)
    if args.fix_length != "":
        args.split = 'test'
        eval_ds = load_dataset(args, dataset_name=args.dataset_name)
        eval_dl = torch.utils.data.DataLoader(
            eval_ds, batch_size=args.batch_size if args.batch_size != -1 else len(eval_ds), 
            shuffle=False
        )

    # 4. generate dataset
    x0_gen_path = f'{args.save_dir}/x0_gen-{args.exp_name}.pt'
    if os.path.exists(x0_gen_path):
        xt = torch.load(x0_gen_path, map_location='cpu')
    else:
        model.to(args.device)
        if args.sampling_schedule_name != "uniform":
            sampling_schedule = torch.load(args.sampling_schedule_path)
            sampling_schedule = list(filter(lambda sampling_schedule: len(sampling_schedule)==args.tgt_nfe+1, sampling_schedule))[0]
            print(f'sampling_schedule-{args.sampling_schedule_name}-{args.tgt_nfe}:', sampling_schedule)
        else:
            sampling_schedule = None

        if args.fix_length != 0:
            xt = []
            for seed, (x0, _) in tqdm(enumerate(eval_dl)):
                xt_i = ddms_sampling_fn(
                    model, scheduler, args.device, tgt_nfe=args.tgt_nfe, src_nfe=args.src_nfe, seed=seed,
                    num_samples=args.cond_num_samples, sample_eps=1e-5, sampling_schedule=sampling_schedule, 
                    fix_length=args.fix_length, max_length=args.max_length, x0=x0,
                )[0].cpu()
                xt.append(xt_i)
            xt = torch.cat(xt, dim=0)
        else:
            xt = [
                ddms_sampling_fn(
                    model, scheduler, args.device, tgt_nfe=args.tgt_nfe, src_nfe=args.src_nfe, seed=seed,
                    num_samples=args.batch_size, sample_eps=1e-5, sampling_schedule=sampling_schedule, 
                )[0].cpu()
                for seed in tqdm(range(args.num_samples // args.batch_size))
            ]
            xt = torch.cat(xt, dim=0)
        torch.save(xt, x0_gen_path)
        del model, scheduler
        flush()

    # 5. evaluate the result
    x0_eval_path = f'{args.output_dir}/x0_eval-{args.exp_name}.pt'
    if os.path.exists(x0_eval_path):
        metric = torch.load(x0_eval_path)
    else:
        eval_fn = load_evaluation_fn(args)
        metric = eval_fn(xt, device=args.device, batch_size=args.batch_size, x0_gen_path=x0_gen_path)
        torch.save(metric, x0_eval_path)
    print(f'{args.exp_name} / Metric: {metric}')