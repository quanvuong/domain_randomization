import numpy as np
import torch
import torch.nn.functional as F

from dr.ppo.utils import Dataset, change_lr


def update_params(m_b, pol, val, optims, clip_param):
    keys = ('obs', 'acs', 'vtargs', 'atargs', 'pold')
    obs, acs, vtargs, atargs, pold = (torch.from_numpy(m_b[i]).float() for i in keys)

    vtargs = vtargs.view(-1, 1)
    atargs = atargs.view(-1, 1)

    # Calculate policy surrogate objective
    pnew = pol.prob(obs, acs)

    ratio = pnew / pold

    surr1 = ratio * atargs
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * atargs
    pol_surr, _ = torch.min(torch.cat((surr1, surr2), dim=1), dim=1)
    pol_surr = - torch.sum(pol_surr) / obs.size()[0]

    # Calculate value function loss
    val_loss = F.mse_loss(val(obs), vtargs)

    optims['pol_optim'].zero_grad()
    optims['val_optim'].zero_grad()

    total_loss = pol_surr + val_loss
    total_loss.backward(retain_graph=True)

    optims['pol_optim'].step()
    optims['val_optim'].step()


@torch.no_grad()
def evaluate_policy(pol, eval_envs):
    num_envs = len(eval_envs)

    for i, env in enumerate(eval_envs):
        env.seed(i)

    done = np.array([False] * num_envs)
    avg_reward = np.array([0.] * num_envs, dtype=np.float32)

    obs = np.stack([env.reset() for env in eval_envs])

    while not all(done):
        t_obs = torch.from_numpy(obs).float()
        _, mean_acs = pol(t_obs)
        for i, (env, action) in enumerate(zip(eval_envs, mean_acs)):
            if not done[i]:
                obs[i], r, d, _ = env.step(action)
                avg_reward[i] += r
                done[i] = d

    avg_reward = np.mean(avg_reward)

    return avg_reward


def add_vtarg_and_adv(seg, lam, gamma):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """

    news = np.append(seg["news"],
                     0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpreds = np.append(seg["vpreds"], seg["nextvpred"])
    T = len(seg["rews"])
    seg["advs"] = gaelam = np.empty(T, 'float32')
    rews = seg["rews"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - news[t + 1]
        delta = rews[t] + gamma * vpreds[t + 1] * nonterminal - vpreds[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamrets"] = seg["advs"] + seg["vpreds"]


def one_train_iter(pol, val, optims,
                   iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                   state_running_m_std, train_params, eval_envs, eval_perfs,
                   eval_freq=5):
    # Extract params
    ts_per_batch = train_params['ts_per_batch']
    num_timesteps = train_params['num_timesteps']
    optim_stepsize = train_params['optim_stepsize']
    lam = train_params['lam']
    gamma = train_params['gamma']
    optim_epoch = train_params['optim_epoch']
    optim_batch_size = train_params['optim_batch_size']
    clip_param = train_params['clip_param']

    # Anneal the learning rate
    num_ts_so_far = iter_i * ts_per_batch
    lr_mult = max(1.0 - float(num_ts_so_far) / num_timesteps, 0)

    change_lr(optims['pol_optim'], optim_stepsize * lr_mult)
    change_lr(optims['val_optim'], optim_stepsize * lr_mult)

    # Obtain training batch
    seg = seg_gen.__next__()

    # Update running mean and std of states
    state_running_m_std.update(seg['obs'])

    eps_rets_buff.extend(seg['ep_rets'])
    eps_rets_mean_buff.append((num_ts_so_far, np.mean(eps_rets_buff)))

    add_vtarg_and_adv(seg, lam, gamma)

    seg['advs'] = (seg['advs'] - seg['advs'].mean()) / seg['advs'].std()
    pold = pol.prob(torch.from_numpy(seg['obs']).float(),
                    torch.from_numpy(seg['acs']).float()).data.numpy()

    batch = Dataset(dict(obs=seg['obs'], acs=seg['acs'], atargs=seg['advs'], vtargs=seg['tdlamrets'], pold=pold))

    for epoch_i in range(optim_epoch):
        for m_b in batch.iterate_once(optim_batch_size):
            update_params(m_b, pol, val, optims, clip_param)

    if iter_i % eval_freq == 0:
        eval_pref = evaluate_policy(pol, eval_envs)

        eval_perfs.append(eval_pref)
