import logging
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import minsr_solve, patches_to_spins, ratio_clip


def evaluate(args, model, ham, beta, k=20):
    f_list = []
    f_std_list = []
    e_list = []
    s_list = []

    for _ in range(k):
        x = model.sample(args.batch_size)
        if args.nn == "transformer" and args.patch_size > 1:  # convert patches to spins
            s = patches_to_spins(x, args.patch_size) * 2.0 - 1.0
        else:
            s = x * 2.0 - 1.0
        energy = ham.energy(s)
        log_probs = model(x)
        loss = log_probs / beta + energy
        free_energy_ = loss.mean().item() / args.n
        free_energy_std_ = loss.std().item() / args.n
        energy_ = energy.mean().item() / args.n
        entropy_ = -1.0 * log_probs.mean().item() / args.n
        f_list.append(free_energy_)
        f_std_list.append(free_energy_std_)
        e_list.append(energy_)
        s_list.append(entropy_)

    free_energy = np.mean(f_list)
    free_energy_std = np.mean(f_std_list)
    energy = np.mean(e_list)
    entropy = np.mean(s_list)

    print(f"Results: f={free_energy:.6g}, f_std={free_energy_std:.6g}\n")
    with open(os.path.join(args.path, "output.txt"), "a", newline="\n") as f:
        f.write(f"{beta:.2f} {free_energy} {free_energy_std} {energy} {entropy}\n")


def train_sgd(args, model, optimizer, ham, beta, writer):
    t1 = time.time()

    epochs = round(args.epochs // args.resample_steps)
    #pbar = tqdm(range(epochs))
    
    ratios = []
    ratios_noclip = []
    rewards = []
    
    for n_iter in range(epochs):
        with torch.no_grad():
            x_old = model.sample(args.batch_size)
            if args.nn == "transformer" and args.patch_size > 1:  # convert patches to spins
                s_old = patches_to_spins(x_old, args.patch_size) * 2.0 - 1.0
            else:
                s_old = x_old * 2.0 - 1.0
            energy = ham.energy(s_old)
            log_probs_old = model(x_old)
            loss_old = log_probs_old / beta + energy

        for i in range(args.resample_steps):
            optimizer.zero_grad()
            log_probs = model(x_old)
            with torch.no_grad():
                
                loss = log_probs / beta + energy
                if not args.forward_KL:
                    ratio = (log_probs - log_probs_old).exp() + 1e-40
                else:
                    ratio = ((-beta * energy) - log_probs).exp()
                
                if args.clip_type != 'none':
                    
                    ###########################################
                    
                    if args.ratio_store and (equal(beta,0.1) or equal(beta,0.2) or equal(beta,0.3) or equal(beta,0.5) or equal(beta,0.8)):
                        if (n_iter == 0 or n_iter == 1 or n_iter == 2 or n_iter == 5 or n_iter == 8 or n_iter == 30 or n_iter == 80):
                            ratios_noclip.append(ratio.clone().cpu())
                    
                    ###########################################
                    
                    ratio = ratio_clip(ratio, args.clip_type, args.clip_index, loss_old - loss, args.clip_penalty, args.drop_clip)
                
                if args.ratio_type == "normal":
                    pass
                elif args.ratio_type == "softmax":
                    ratio = torch.nn.functional.softmax(ratio, dim=0)
                elif args.ratio_type == "tanh":
                    ratio = torch.nn.functional.tanh(ratio)
                elif args.ratio_type == "softmaxlog":
                    ratio = torch.nn.functional.softmax(torch.log(ratio), dim=0)
                else:
                    raise ValueError(f"Unknown ratio type {args.ratio_type}")
                
            if not args.forward_KL:
                temp = args.scale_index * ratio * log_probs * (loss - loss.mean())
                loss_reinforce = temp[temp != 0].float().mean()
                #loss_reinforce = torch.mean(args.scale_index * ratio * log_probs * (loss - loss.mean()))
            else:
                loss_reinforce = -torch.mean(args.scale_index * ratio * log_probs)
            loss_reinforce.backward()
            optimizer.step()
            
            ########################
            if args.ratio_store and (n_iter <= 10 or n_iter % 10) == 0:
                with open(args.path + f'/ratio_{args.ratio_type}.log', 'a', newline='\n') as f:
                    f.write(f'beta: {beta}, epoch: {n_iter}, resample_step: {i}, mean ratio: {ratio.mean()}, max ratio: {max(ratio)}, min ratio: {min(ratio)}, loss: {loss_reinforce}' + u'\n')
            
            if args.ratio_store and (equal(beta,0.1) or equal(beta,0.2) or equal(beta,0.3) or equal(beta,0.5) or equal(beta,0.8)):
                if (n_iter == 0 or n_iter == 1 or n_iter == 2 or n_iter == 5 or n_iter == 8 or n_iter == 30 or n_iter == 80):
                    ratios.append(ratio.clone().cpu())
                    rewards.append(((loss.mean().item() / args.n) - (loss_old.mean().item() / args.n)))
            ########################

    if ratios:
        ratios = torch.stack(ratios)
        np.savetxt(args.path + f'/ratios_{beta:.1f}.txt', ratios)
        
    if rewards:
        #rewards = torch.stack(rewards)
        np.savetxt(args.path + f'/rewards_{beta:.1f}.txt', rewards)
        
    if ratios_noclip and args.clip_type != 'none':
        ratios_noclip = torch.stack(ratios_noclip)
        np.savetxt(args.path + f'/ratios_noclip_{beta:.1f}.txt', ratios_noclip)
    
    t_train = time.time() - t1
    evaluate(args, model, ham, beta)
    logging.info(f"beta: {beta:.2f}, running time: {t_train:.2f}s")


def train_ng(args, model, ham, beta, writer):
    t1 = time.time()

    epochs = round(args.epochs // args.resample_steps)
    pbar = tqdm(range(epochs))
    for n_iter in pbar:
        with torch.no_grad():
            x_old = model.sample(args.batch_size)
            if args.nn == "transformer" and args.patch_size > 1:  # convert patches to spins
                s_old = patches_to_spins(x_old, args.patch_size) * 2.0 - 1.0
            else:
                s_old = x_old * 2.0 - 1.0
            energy = ham.energy(s_old)
            log_probs_old = model(x_old)

        for _ in range(args.resample_steps):
            grads = model.per_sample_grad(x_old)  # d logP(x_i) / d theta_j, dict
            grads_flatten = torch.cat([torch.flatten(v, start_dim=1) for v in grads.values()], dim=1)  # N x M
            O_mat = grads_flatten / math.sqrt(args.batch_size)
            log_probs = model(x_old)
            ratio = (log_probs - log_probs_old).exp()
            loss = log_probs / beta + energy
            R_vec = (ratio * (loss - loss.mean())) / math.sqrt(args.batch_size)
            O_mat, R_vec = O_mat.double(), R_vec.double()
            dtheta_flatten = minsr_solve(O_mat, R_vec, lambd=args.lambd)
            if args.adaptive_lr:
                # lr = math.sqrt(2 * args.lr / (torch.dot(O_mat.T @ R_vec, dtheta_flatten)))  # given by lagrange multiplier
                lr = args.lr * 0.998**n_iter  # annealing schedule, 0.98^100 ~ 0.13, 0.998^100 ~ 0.13
            else:
                lr = args.lr
            model.update_params(dtheta_flatten.float(), lr)

        with torch.no_grad():
            free_energy_ = loss.mean().item() / args.n
            free_energy_std_ = loss.std().item() / args.n
            energy_ = energy.mean().item() / args.n
            entropy_ = -1.0 * log_probs.mean().item() / args.n
            pbar.set_description(f"beta: {beta:.2f}, f: {free_energy_:.4g}, f_std: {free_energy_std_:.4g}")

        if args.use_tb:
            writer.add_scalar(f"beta{beta:.2f}/free_energy", free_energy_, n_iter)
            writer.add_scalar(f"beta{beta:.2f}/free_energy_std", free_energy_std_, n_iter)
            writer.add_scalar(f"beta{beta:.2f}/energy", energy_, n_iter)
            writer.add_scalar(f"beta{beta:.2f}/entropy", entropy_, n_iter)

    t_train = time.time() - t1
    evaluate(args, model, ham, beta)
    logging.info(f"beta: {beta:.2f}, running time: {t_train:.2f}s")

def equal(num1,num2):
    return abs(num1 - num2) <= 1e-10