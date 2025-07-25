#!/usr/bin/env python3
#
# Solving Statistical Mechanics using Variational Autoregressive Networks
# 2d classical Ising model

import time

import numpy as np
import torch
import torchvision
from numpy import sqrt
from torch import nn

import ising
from args import args
from bernoulli import BernoulliMixture
from made import MADE
from pixelcnn import PixelCNN
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
    ratio_clip,
    process_data,
)


def main():
    start_time = time.time()

    init_out_dir()
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    print_args()

    if args.net == 'made':
        net = MADE(**vars(args))
    elif args.net == 'pixelcnn':
        net = PixelCNN(**vars(args))
    elif args.net == 'bernoulli':
        net = BernoulliMixture(**vars(args))
    else:
        raise ValueError('Unknown net: {}'.format(args.net))
    net.to(args.device)
    my_log('{}\n'.format(net))

    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    named_params = list(net.named_parameters())

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

    if args.lr_schedule:
        # 0.92**80 ~ 1e-3
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.92, patience=100, threshold=1e-4, min_lr=1e-6)

    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename,
                                                     last_step))
        ignore_param(state['net'], net)
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
        if args.lr_schedule and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])

    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))

    my_log('Training...')
    sample_time = 0
    train_time = 0
    start_time = time.time()
    optimize_step = 0
    log_prob_old = 0
    data_collector = {'free_energy_mean':0, 'free_energy_std':0, 'beta':0, 'beta_anneal':args.beta_anneal, 'time':0, 'steps':args.importance_sampling_steps, 'epsilon': args.importance_sampling_epsilon}
    early_stop_steps = 0
    last_free_energy = 0
    for step in range(last_step + 1, args.max_step + 1):
        
        if early_stop_steps >= 10:
            break
        
        if optimize_step == 0:
            sample_start_time = time.time()
            with torch.no_grad():
                sample, x_hat = net.sample(args.batch_size)
            assert not sample.requires_grad
            assert not x_hat.requires_grad
            sample_time += time.time() - sample_start_time
            log_prob_old = net.log_prob(sample)
            
            # 0.998**9000 ~ 1e-8
            beta = args.beta * (1 - args.beta_anneal**(step + 1))
            
        optimizer.zero_grad()

        train_start_time = time.time()
        
        #log_prob shape: (batch,)
        log_prob = net.log_prob(sample)
        
        with torch.no_grad():
            energy = ising.energy(sample, args.ham, args.lattice,
                                  args.boundary)
            loss = log_prob + beta * energy
            ratio = (log_prob - log_prob_old).exp() + 1e-40
        assert not energy.requires_grad
        assert not loss.requires_grad
        reward = loss - loss.mean()

        if args.clip_type != 'none':
            ratio = ratio_clip(ratio, args.clip_type, args.importance_sampling_epsilon)

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

        loss_reinforce = torch.mean(reward * log_prob * ratio)
        loss_reinforce.backward()

        if args.clip_grad:
            nn.utils.clip_grad_norm_(params, args.clip_grad)

        optimizer.step()
        
        optimize_step = (optimize_step + 1) % args.importance_sampling_steps

        if args.lr_schedule:
            scheduler.step(loss.mean())

        train_time += time.time() - train_start_time

        if args.print_step and step % args.print_step == 0:
            free_energy_mean = loss.mean() / args.beta / args.L**2
            
            if abs(free_energy_mean - last_free_energy) <= 1e-8:
                early_stop_steps += 1
            else:
                early_stop_steps = 0
            last_free_energy = free_energy_mean
            
            free_energy_std = loss.std() / args.beta / args.L**2
            entropy_mean = -log_prob.mean() / args.L**2
            energy_mean = energy.mean() / args.L**2
            mag = sample.mean(dim=0)
            mag_mean = mag.mean()
            mag_sqr_mean = (mag**2).mean()
            if step > 0:
                sample_time /= args.print_step
                train_time /= args.print_step
            used_time = time.time() - start_time
            my_log(
                'step = {}, F = {:.8g}, F_std = {:.8g}, prob = {:.8g}, prob_old = {:.8g}, ratio = {:.8g}, |reward * log_prob| * prob = {:.8g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                .format(
                    step,
                    free_energy_mean.item(),
                    free_energy_std.item(),
                    log_prob.exp().mean(),
                    log_prob_old.exp().mean(),
                    ratio.mean(),
                    (abs(reward * log_prob) * log_prob.exp()).mean(),
                    beta,
                    sample_time,
                    train_time,
                    used_time,
                ))
            sample_time = 0
            train_time = 0

            if args.save_sample:
                state = {
                    'sample': sample,
                    'x_hat': x_hat,
                    'log_prob': log_prob,
                    'energy': energy,
                    'loss': loss,
                }
                torch.save(state, '{}_save/{}.sample'.format(
                    args.out_filename, step))

        if (args.out_filename and args.save_step
                and step % args.save_step == 0):
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if args.lr_schedule:
                state['scheduler'] = scheduler.state_dict()
            torch.save(state, '{}_save/{}.state'.format(
                args.out_filename, step))

        if (args.out_filename and args.visual_step
                and step % args.visual_step == 0):
            torchvision.utils.save_image(
                sample,
                '{}_img/{}.png'.format(args.out_filename, step),
                nrow=int(sqrt(sample.shape[0])),
                padding=0,
                normalize=True)

            if args.print_sample:
                x_hat_np = x_hat.view(x_hat.shape[0], -1).cpu().numpy()
                x_hat_std = np.std(x_hat_np, axis=0).reshape([args.L] * 2)

                x_hat_cov = np.cov(x_hat_np.T)
                x_hat_cov_diag = np.diag(x_hat_cov)
                x_hat_corr = x_hat_cov / (
                    sqrt(x_hat_cov_diag[:, None] * x_hat_cov_diag[None, :]) +
                    args.epsilon)
                x_hat_corr = np.tril(x_hat_corr, -1)
                x_hat_corr = np.max(np.abs(x_hat_corr), axis=1)
                x_hat_corr = x_hat_corr.reshape([args.L] * 2)

                energy_np = energy.cpu().numpy()
                energy_count = np.stack(
                    np.unique(energy_np, return_counts=True)).T

                my_log(
                    '\nsample\n{}\nx_hat\n{}\nlog_prob\n{}\nenergy\n{}\nloss\n{}\nx_hat_std\n{}\nx_hat_corr\n{}\nenergy_count\n{}\n'
                    .format(
                        sample[:args.print_sample, 0],
                        x_hat[:args.print_sample, 0],
                        log_prob[:args.print_sample],
                        energy[:args.print_sample],
                        loss[:args.print_sample],
                        x_hat_std,
                        x_hat_corr,
                        energy_count,
                    ))

            if args.print_grad:
                my_log('grad max_abs min_abs mean std')
                for name, param in named_params:
                    if param.grad is not None:
                        grad = param.grad
                        grad_abs = torch.abs(grad)
                        my_log('{} {:.3g} {:.3g} {:.3g} {:.3g}'.format(
                            name,
                            torch.max(grad_abs).item(),
                            torch.min(grad_abs).item(),
                            torch.mean(grad).item(),
                            torch.std(grad).item(),
                        ))
                    else:
                        my_log('{} None'.format(name))
                my_log('')

        if (args.max_step - step < 10 or early_stop_steps >= 10):
            free_energy_mean = loss.mean() / args.beta / args.L**2
            free_energy_std = loss.std() / args.beta / args.L**2
            data_collector['free_energy_mean'] += free_energy_mean
            data_collector['free_energy_std'] += free_energy_std
            if (args.max_step == step or early_stop_steps >= 10):
                used_time = time.time() - start_time
                data_collector['free_energy_mean'] /= 10
                data_collector['free_energy_std'] /= 10
                data_collector['beta'] = beta
                data_collector['time'] = used_time

    process_data(data_collector, args.name)

if __name__ == '__main__':
    main()
