import os 
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm 
from copy import deepcopy

from utils import preset
from dataset import load_dataset, load_validation_fn
from model import DDiT, DDMLP
from ddms import d3pm, sedd

class EMA:
    def __init__(self, model, ema_decay):
        self.ema_decay = ema_decay
        self.ema_param = [p.clone().detach().requires_grad_(False) for p in model.parameters() if p.requires_grad]
        self.num_updates = 0

    def to(self, device, dtype):
        self.ema_param = [p.to(device, dtype) for p in self.ema_param]

    def update_ema(self, model):
        if len(self.ema_param) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        self.num_updates += 1
        ema_decay = min(self.ema_decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - ema_decay
        with torch.no_grad():
            parameters = [p for p in model.parameters() if p.requires_grad]
            for ema_param, param in zip(self.ema_param, parameters):
                ema_param.sub_(one_minus_decay * (ema_param - param))

if __name__ == "__main__":
    # 1. config
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["moons", "count", "cifar", "piano"])
    parser.add_argument("--model_name", type=str, choices=["d3pm", "sedd"])
    parser.add_argument("--arch_name", type=str, choices=["ddit", "ddmlp"])
    parser.add_argument("--noise_schedule", type=str, default="loglinear")
    parser.add_argument("--scheduler_name", type=str, default="tweedie", choices=["euler", "tweedie"])
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--dataset_path", type=str, default='../datasets')
    parser.add_argument("--wandb_key", type=str, default='')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args = preset(args)
    args.output_dir = f"runs/{args.dataset_name}-{args.model_name}"
    args.exp = f"{args.lr}-{args.batch_size}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    dtype = torch.float32
    eps = args.eps

    # 2. load dataset, dataloader
    ds = load_dataset(args)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)

    # 3. load model 
    if args.arch_name == "ddit":
        model = DDiT(
            num_vocabs=args.num_vocabs,
            cond_dim=args.cond_dim,
            hidden_size=args.hidden_size,
            n_heads=args.n_heads,
            dropout=args.dropout,
            n_blocks=args.n_blocks,
            mlp_ratio=args.mlp_ratio,
        )
    if args.arch_name == "ddmlp":
        model = DDMLP(
            num_vocabs=args.num_vocabs,
            length=args.length,
            hidden_size=args.hidden_size,
            n_blocks=args.n_blocks,
        )
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    # 4. load scheduler, loss function
    if args.model_name == "d3pm":
        if args.scheduler_name == "euler":
            scheduler = d3pm.EulerScheduler(args)
        if args.scheduler_name == "tweedie":
            scheduler = d3pm.AnalyticScheduler(args)
        loss_fn = d3pm.Loss(scheduler)
    if args.model_name == "sedd":
        if args.scheduler_name == "euler":
            scheduler = sedd.EulerScheduler(args)
        if args.scheduler_name == "tweedie":
            scheduler = sedd.AnalyticScheduler(args)
        loss_fn = sedd.Loss(scheduler)

    # o. logging
    import wandb
    if args.wandb_key != "":
        wandb.login(
            key=args.wandb_key
        )
    wandb.init(
        project=f"{args.dataset_name}-{args.model_name}",
        name=args.exp,
        config={} # Track hyperparameters and run metadata
    )

    # 5. prepare training 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, args.ema_decay)

    # 6. train
    ema.to(device, dtype)
    model.to(device, dtype)
    scheduler.to(device, dtype)

    loss_traj = []
    global_step = 0
    for _ in range(int(1e8)):
        model.train()
        for x0, _ in dl:
            time_s = time.time()
            x0 = x0.to(device)

            # perturb x0
            t = (1 - eps) * torch.rand(x0.shape[0], device=x0.device) + eps
            xt = scheduler.add_noise(x0, t)
            
            # model forward
            sigma_bar = scheduler.sigma_bar(t)
            output = model(xt, sigma_bar)
            
            # compute loss function 
            loss = loss_fn(output, sigma_bar, xt, x0)
            assert not loss.isnan().any()
            
            # update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * np.minimum(global_step / args.warmup, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update_ema(model)

            loss_traj.append(loss.item())
            wandb.log({'loss': loss.item()}, step=global_step)
            global_step += 1

            time_e = time.time()
                
            if args.train_iter == global_step:
                print(f'iter: {round(time_e - time_s, 5)}sec')
                torch.save(ema.ema_param, f"{args.output_dir}/ema_param-{args.exp}.pt")
                torch.save(model.state_dict(), f"{args.output_dir}/state_dict-{args.exp}.pt")
                assert False, 'finish training'

    # 7. validation 
    # validation_fn = load_validation_fn(args)
    # validation_fn(
    #     model, scheduler, device, x0, path=f"{args.output_dir}/x0_gen-{args.scheduler_name}",
    #     num_samples=512, num_function_eval=1024, sample_eps=1e-3, 
    # )