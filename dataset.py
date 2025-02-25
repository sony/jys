import random
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from copy import deepcopy
from tqdm import tqdm

#########
# Moons #
#########
class MoonsDataset:
    def __init__(self, n_samples=1024, tokens=256, dtype=torch.long):
        from sklearn.datasets import make_moons
        x, _ = make_moons(n_samples=n_samples, shuffle=True, noise=None, random_state=None)
        x = x - x.min()
        x = x / x.max()
        x = (tokens * x).astype(int)
        assert x.max() <= tokens and x.min() >= 0
        self.x = torch.from_numpy(x).to(dtype)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.size(0), torch.tensor(0)

def moons_validation_fn(
    model, scheduler, device, x0, path, 
    num_samples=512, num_function_eval=2048, sample_eps=1e-3, 
):
    xt = scheduler.sample_latent(num_samples).to(device)
    timesteps = torch.linspace(1, sample_eps, num_function_eval + 1, device=device)

    for i in tqdm(range(num_function_eval), desc='eval...'):
        dt = timesteps[i] - timesteps[i+1]
        t = timesteps[i] * torch.ones(xt.shape[0], device=device)

        sigma_bar = scheduler.sigma_bar(t)
        output = model(xt, sigma_bar)
        xt = scheduler.step(output, xt, t, dt).xt

    x0_cpu = x0.detach().cpu()
    xt_cpu = xt.detach().cpu()

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(x0_cpu[:, 0], x0_cpu[:, 1])
    axs[1].scatter(xt_cpu[:, 0], xt_cpu[:, 1])
    plt.savefig(f"{path}.jpg")
    plt.close()
    return 

#############
# CountDown #
#############
class CountDataset:
    def __init__(self, num_samples=2048, num_vocabs=32, length=256):
        self.num_samples = num_samples
        self.num_vocabs = num_vocabs
        self.length = length
        self.x0 = torch.tensor(
            [self.generate_countdown_sample(length, num_vocabs-1) for _ in range(num_samples)]
        )

    def __getitem__(self, index):
        return self.x0[index], torch.tensor(0)

    def __len__(self):
        return self.num_samples

    def generate_countdown_sample(self, length, max_number):
        sample = []
        remaining_length = length

        while remaining_length > 0:
            # Randomly choose a starting number
            start_number = random.randint(0, max_number)

            # Generate countdown sequence
            countdown = list(range(start_number, -1, -1))

            # Add countdown to sample
            sample.extend(countdown)

            # Update remaining length
            remaining_length -= len(countdown)

        # Truncate exceed tokens
        if len(sample) > length:
            sample = sample[:length]

        assert len(sample) == length
        assert all(check_err(deepcopy(sample), [], None))
        return sample

def check_err(sample, err, prev_number=None):
    assert isinstance(sample, list)

    if len(sample) == 0:
        return err
    
    elif prev_number is None:
        prev_number = sample.pop(0)
        err.append(True)
        return check_err(sample, err, prev_number)
    
    else:
        curr_number = sample.pop(0)
        if prev_number == 0:
            err.append(True)
        else:
            err.append(prev_number - 1 == curr_number)
            curr_number = 0
        return check_err(sample, err, prev_number=curr_number)
    
def count_validation_fn(
    model, scheduler, device, x0, path, timesteps=None,
    num_samples=512, num_function_eval=2048, sample_eps=1e-3, 
):
    xt = scheduler.sample_latent(num_samples).to(device)
    timesteps = (
        torch.linspace(1, sample_eps, num_function_eval + 1, device=device) 
        if timesteps is None 
        else timesteps
    )

    for i in tqdm(range(num_function_eval), desc='eval...'):
        dt = timesteps[i] - timesteps[i+1]
        t = timesteps[i] * torch.ones(xt.shape[0], device=device)

        sigma_bar = scheduler.sigma_bar(t)
        output = model(xt, sigma_bar)
        xt = scheduler.step(output, xt, t, dt).xt

    x0_cpu = F.one_hot(x0).detach().cpu()
    xt_cpu = F.one_hot(xt).detach().cpu()

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(x0_cpu[0].T)
    axs[1].imshow(xt_cpu[0].T)
    plt.savefig(f"{path}.jpg")
    plt.close()

    prop_of_error = []
    for i in range(num_samples):
        err_list = check_err(xt[i].tolist(), [], prev_number=None)
        prop_of_error.append(1 - np.sum(err_list) / len(err_list))
    prop_of_error = float(np.mean(prop_of_error))
    print('num_function_eval:', round(prop_of_error, 3))
    
    with open(f"{path}.txt", "w+") as f:
        f.write(f"{num_function_eval}-{prop_of_error}")

    return prop_of_error

def count_err_fn(
    xt, device, batch_size=None, **kwargs,
):
    prop_of_error = []
    for xt_i in xt:
        err_list = check_err(xt_i.tolist(), [], prev_number=None)
        prop_of_error.append(1 - np.sum(err_list) / len(err_list))
    prop_of_error = float(np.mean(prop_of_error))
    return prop_of_error

###################
# Text generation #
###################
class WebTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_samples, cache_dir, split=None, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        max_length = 1024
        
        from datasets import load_dataset as load_dataset_hf
        self.dataset = load_dataset_hf(
            "stas/openwebtext-10k", num_proc=8, cache_dir=f'{cache_dir}'
        )["train"]
        self.dataset = self.dataset if split == 'train' else reversed(self.dataset)

        self.input_ids_list = []
        for item in tqdm(self.dataset, desc="tokenzing dataset..."):
            input_ids = self.tokenizer(
                item["text"],
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze()
            if len(input_ids) == max_length:
                self.input_ids_list.append(input_ids)
            else:
                pass

            if len(self.input_ids_list) == self.num_samples:
                break
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_ids_list[idx], torch.tensor(0)
    
class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_samples, cache_dir, max_length=1024):
        from datasets import load_dataset as load_dataset_hf
        self.dataset = load_dataset_hf(
            "openwebtext", num_proc=8, cache_dir=f'{cache_dir}'
        )["train"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples

        self.input_ids_list = []
        for item in tqdm(self.dataset, desc="tokenzing dataset..."):
            try:
                input_ids = self.tokenizer(
                    item["text"],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.squeeze()
                assert len(input_ids) == 1024
                self.input_ids_list.append(input_ids)
            except:
                pass

            if len(self.input_ids_list) == self.num_samples:
                break
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_ids_list[idx], torch.tensor(0)
    
@torch.no_grad()
def ppl_fn(
    input_ids, device, batch_size=8, **kwargs,
):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    nlls = []
    model.to(device)
    for input_ids_i in tqdm(input_ids.split(batch_size)):
        input_ids_i = input_ids_i.to(device)
        target_ids_i = input_ids_i.clone()

        outputs = model(input_ids_i, labels=target_ids_i)
        neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

#########
# Piano #
#########
class PianoDataset:
    def __init__(self, args):
        # num_vocabs = args.num_vocabs # S
        # length = args.length         # L
        np_data = np.load(f'{args.dataset_path}/piano/{args.split}.npy') # (N, L) in range [0, S)
        self.data = torch.from_numpy(np_data).to(args.device)
        self.num_samples = args.num_samples

    def __len__(self):
        if self.num_samples != -1:
            return self.num_samples
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0)

def piano_metric_fn(
    xt, device, ds, fix_length, num_vocabs, *args, **kwargs,
):
    error_dict = {'outlier': [], 'hellinger': []}

    for i in range(len(ds)):
        fake_x0 = xt[i]
        true_x0, _ = ds[i]
        assert fake_x0.shape == true_x0.shape, print(fake_x0.shape, true_x0.shape)
        assert (fake_x0[:fix_length].cpu() == true_x0[:fix_length].cpu()).all()
        fake_x0 = fake_x0[fix_length:].cpu()
        true_x0 = true_x0[fix_length:].cpu()
        fake_hist = F.one_hot(fake_x0, num_classes=num_vocabs).float().mean(dim=0).to(device)
        true_hist = F.one_hot(true_x0, num_classes=num_vocabs).float().mean(dim=0).to(device)

        # Hellinger Distance 
        error_dict['hellinger'].append(hellinger(fake_hist, true_hist).item())

        # Proportion of Outlier 
        error_dict['outlier'].append(fake_hist[true_hist == 0].sum().item())
        
    error_dict['hellinger'] = np.mean(error_dict['hellinger'])
    error_dict['outlier'] = np.mean(error_dict['outlier'])
    return error_dict

_SQRT2 = np.sqrt(2).item()
def hellinger(p, q):
    return torch.sqrt(torch.sum((p.sqrt() - q.sqrt()) ** 2)) / _SQRT2

###########
# CIFAR10 #
###########
import torchvision

class CIFAR10Dataset:
    def __init__(self, args):
        ds = torchvision.datasets.CIFAR10('../datasets', train=True, download=True)
        
        self.x0 = []
        for x, _ in ds:
            self.x0.append(torch.tensor(np.array(x)).flatten(0).long())
            if len(self.x0) == args.num_samples:
                break
        self.num_samples = args.num_samples

    def __getitem__(self, index):
        return self.x0[index], torch.tensor(0)
    
    def __len__(self):
        return self.num_samples
    
def cifar_eval_fn(xt, x0_gen_path, device, **kwargs):
    import torch_fidelity
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=x0_gen_path, input2='cifar10-train', 
        gpu=device, cuda=True, fid=True, prc=False, verbose=False,
    )
    return metrics_dict['frechet_inception_distance']

##########
# loader #
##########
def load_dataset(args, model=None, dataset_name=None):
    dataset_name = args.dataset_name if dataset_name is None else dataset_name 
    
    if dataset_name == "moons":
        ds = MoonsDataset(tokens=args.num_vocabs)
    elif dataset_name == "count":
        ds = CountDataset(num_samples=args.num_samples, length=args.length, num_vocabs=args.num_vocabs)
    elif dataset_name == "piano":
        ds = PianoDataset(args)
    elif dataset_name == "cifar":
        ds = CIFAR10Dataset(args)
    elif dataset_name == "text":
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        ds = WebTextDataset(
            tokenizer=tokenizer, num_samples=args.num_samples, split=args.split,
            cache_dir=args.cache_dir, max_length=1024,
        )
    else:
        raise ValueError()
    return ds

def load_validation_fn(args):
    if args.dataset_name == "moons":
        return moons_validation_fn
    if args.dataset_name == "count":
        return count_validation_fn
    raise ValueError('invalid dataset_name:', args.dataset_name)

def load_evaluation_fn(args):
    if args.dataset_name == "text":
        return ppl_fn
    if args.dataset_name == "count":
        return count_err_fn
    if args.dataset_name == "piano":
        ds = PianoDataset(args)
        piano_eval_fn = partial(piano_metric_fn, ds=ds, fix_length=args.fix_length, num_vocabs=args.num_vocabs)
        return piano_eval_fn
    if args.dataset_name == "cifar":
        return cifar_eval_fn
    raise ValueError('invalid dataset_name:', args.dataset_name)