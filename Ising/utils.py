import os
from glob import glob

import numpy as np
import torch

from args import args

if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True

if not args.seed:
    args.seed = np.random.randint(1, 10**8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')

args.out_filename = None


def get_ham_args_features():
    ham_args = '{ham}_{lattice}_{boundary}_L{L}_beta{beta:g}'
    ham_args = ham_args.format(**vars(args))

    if args.net == 'made':
        features = 'nd{net_depth}_nw{net_width}_made'
    elif args.net == 'bernoulli':
        features = 'bernoulli_nw{net_width}'
    else:
        features = 'nd{net_depth}_nw{net_width}_hks{half_kernel_size}'

    if args.bias:
        features += '_bias'
    if args.z2:
        features += '_z2'
    if args.res_block:
        features += '_res'
    if args.x_hat_clip:
        features += '_xhc{x_hat_clip:g}'
    if args.final_conv:
        features += '_fconv'

    if args.optimizer != 'adam':
        features += '_{optimizer}'
    if args.lr_schedule:
        features += '_lrs'
    if args.beta_anneal:
        features += '_ba{beta_anneal:g}'
    if args.clip_grad:
        features += '_cg{clip_grad:g}'
        
    if args.importance_sampling_steps:
        features += '_is{importance_sampling_steps:g}'
    if args.ratio_type:
        features += '_rt{ratio_type:g}'
    if args.clip_type:
        features += '_ct{clip_type:g}'

    features = features.format(**vars(args))

    return ham_args, features


def init_out_filename():
    if not args.out_dir:
        return
    ham_args, features = get_ham_args_features()
    template = '{args.out_dir}/{ham_args}/{features}/out{args.out_infix}'
    args.out_filename = template.format(**{**globals(), **locals()})


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.out_dir:
        return
    init_out_filename()
    ensure_dir(args.out_filename)
    if args.save_step:
        ensure_dir(args.out_filename + '_save/')
    if args.visual_step:
        ensure_dir(args.out_filename + '_img/')


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def clear_err():
    if args.out_filename:
        open(args.out_filename + '.err', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def my_err(s):
    if args.out_filename:
        with open(args.out_filename + '.err', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        print_fn('{} = {}'.format(k, v))
    print_fn('')


def parse_checkpoint_name(filename):
    filename = os.path.basename(filename)
    filename = filename.replace('.state', '')
    step = int(filename)
    return step


def get_last_checkpoint_step():
    if not (args.out_filename and args.save_step):
        return -1
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    if not filename_list:
        return -1
    step = max([parse_checkpoint_name(x) for x in filename_list])
    return step


def clear_checkpoint():
    if not (args.out_filename and args.save_step):
        return
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    for filename in filename_list:
        os.remove(filename)


# Do not load some params
def ignore_param(state, net):
    ignore_param_name_list = ['x_hat_mask', 'x_hat_bias']
    param_name_list = list(state.keys())
    for x in param_name_list:
        for y in ignore_param_name_list:
            if y in x:
                state[x] = net.state_dict()[x]
                break

def ratio_clip(ratio, type, epsilon):
    mean_ratio = ratio.mean()
    ranked_ratio = ratio.clone().sort().values
    n = len(ranked_ratio)
    
    if type == 'mean':
        max_ratio = torch.full_like(ratio, mean_ratio*(1+epsilon))
        min_ratio = torch.full_like(ratio, mean_ratio*(1-epsilon))
    elif type == 'meanupper':
        max_ratio = torch.full_like(ratio, mean_ratio*(1+epsilon))
        min_ratio = torch.full_like(ratio, ranked_ratio[0])
    elif type == 'meanlower':
        max_ratio = torch.full_like(ratio, ranked_ratio[-1])
        min_ratio = torch.full_like(ratio, mean_ratio*(1-epsilon))
    elif type == 'dist':
        max_ratio = torch.full_like(ratio, ranked_ratio[int(-n * epsilon)])
        min_ratio = torch.full_like(ratio, ranked_ratio[int(n * epsilon)])
    elif type == 'distupper':
        max_ratio = torch.full_like(ratio, ranked_ratio[int(-n * epsilon)])
        min_ratio = torch.full_like(ratio, ranked_ratio[0])
    elif type == 'distlower':
        max_ratio = torch.full_like(ratio, ranked_ratio[-1])
        min_ratio = torch.full_like(ratio, ranked_ratio[int(n * epsilon)])
    else:
        raise ValueError(f'unknown clip type: {type}')
    
    # if penalty != 'none':
    #     output = torch.logical_or((ratio <= max_ratio), (reward > 0)).float() * ratio + torch.logical_and((ratio > max_ratio), (reward < 0)).float() * max_ratio
    #     output = torch.logical_or((output >= min_ratio), (reward < 0)).float() * output + torch.logical_and((output < min_ratio), (reward > 0)).float() * min_ratio
    # else:
    #     output = (ratio >= min_ratio).float() * ratio + (ratio < min_ratio).float() * min_ratio
    #     output = (output <= max_ratio).float() * output + (output > max_ratio).float() * max_ratio

    output = (ratio >= min_ratio).float() * ratio + (ratio < min_ratio).float() * min_ratio
    output = (output <= max_ratio).float() * output + (output > max_ratio).float() * max_ratio

    output.requires_grad = False
    return output

def process_data(data, name):
    if args.out_filename:
        with open(f'out/{name}_data.text', 'a', newline='\n') as f:
            f.write('{beta} {beta_anneal} {steps} {epsilon} {free_energy_mean} {free_energy_std} {time}'
                    .format(**data) 
                    + u'\n')