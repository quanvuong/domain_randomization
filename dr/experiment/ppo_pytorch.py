import random
from collections import deque
from datetime import datetime

import gym
import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim
from mpi4py import MPI

import os
import os.path as osp
import pickle
import dr
from dr.ppo.models import Policy, ValueNet
from dr.ppo.train import one_train_iter
from dr.ppo.utils import set_torch_num_threads, RunningMeanStd, traj_seg_gen

COMM = MPI.COMM_WORLD
import tensorboardX


def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)


class CEMOptimizer(object):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25, viz_dir=None):
        """Creates an instance of this class.
        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function

        if viz_dir is not None:
            self.writer = tensorboardX.SummaryWriter(viz_dir)
        else:
            self.writer = tensorboardX.SummaryWriter()

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        costs_hist = []
        mean_hist = []
        var_hist = []

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)

            costs = self.cost_function(samples, t)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            for i, m in enumerate(mean):
                self.writer.add_scalar(f'mean/{i}', m, t)

            for i, m in enumerate(var):
                self.writer.add_scalar(f'var/{i}', m, t)

            self.writer.add_scalar('costs', np.min(costs), t)

            t += 1

            costs_hist.append(costs)
            mean_hist.append(mean)
            var_hist.append(var)

        self.writer.close()

        return dict(
            mean_hist=mean_hist, costs_hist=costs_hist, var_hist=var_hist
        )


class PPO_Pytorch(object):

    def __init__(self, experiment_name, env_params, train_params, **kwargs):
        self.experiment_name = experiment_name
        self.env_params = env_params
        self.train_params = train_params

        self.log_dir = osp.join('runs',
                                f'seed_{str(train_params["seed"])}_{datetime.now().strftime("%b%d_%H-%M-%S")}')

        os.makedirs(self.log_dir, exist_ok=True)

        with open(osp.join(self.log_dir, 'env_params.pkl'), 'wb+') as f:
            pickle.dump(env_params, f)

        with open(osp.join(self.log_dir, 'train_params.pkl'), 'wb+') as f:
            pickle.dump(train_params, f)

        super().__init__()

    def train(self, env_id, backend,
              train_params, env_params,
              means, stdevs):

        # Unpack params
        hid_size = train_params['hid_size']
        pol_init_std = train_params['pol_init_std']
        adam_epsilon = train_params['adam_epsilon']
        optim_stepsize = train_params['optim_stepsize']

        # Convert means and stdevs to dict format
        assert len(means) == len(stdevs), (len(means), len(stdevs))
        mean_dict, stdev_dict = PPO_Pytorch._vec_to_dict(env_id, means, stdevs)

        # Set parameter of env
        self.env_dist.default_parameters = mean_dict
        self.env_dist.stdev_dict = stdev_dict
        env = self.env_dist.root_env

        set_torch_num_threads()

        # Construct policy and value network
        pol = Policy(env.observation_space, env.action_space, hid_size, pol_init_std)
        pol_optim = optim.Adam(pol.parameters(), lr=optim_stepsize, eps=adam_epsilon)

        val = ValueNet(env.observation_space, hid_size)
        val_optim = optim.Adam(val.parameters(), lr=optim_stepsize, eps=adam_epsilon)

        optims = {'pol_optim': pol_optim, 'val_optim': val_optim}

        num_train_iter = int(train_params['num_timesteps'] / train_params['ts_per_batch'])

        # Buffer for running statistics
        eps_rets_buff = deque(maxlen=100)
        eps_rets_mean_buff = []

        state_running_m_std = RunningMeanStd(shape=env.observation_space.shape)

        # seg_gen is a generator that yields the training data points
        seg_gen = traj_seg_gen(self.env_dist, pol, val, state_running_m_std, env_params, train_params)

        eval_perfs = []

        for iter_i in range(num_train_iter):
            one_train_iter(pol, val, optims,
                           iter_i, eps_rets_buff, eps_rets_mean_buff, seg_gen,
                           state_running_m_std, train_params, self.eval_envs, eval_perfs)

        return eval_perfs

    def run(self):
        set_global_seeds(self.train_params['seed'])

        # Unpack params
        env_name = self.env_params['env_name']
        backend = self.env_params['backend']

        stdev = self.train_params['env_dist_stdev']
        mean_scale = self.train_params['mean_scale']
        seed = self.train_params['seed']

        num_eval_env = self.train_params['num_eval_env']
        collision_detector = self.env_params['collision_detector']

        # Obtain the initial value for the simulation parameters
        env_dist = dr.dist.Normal(env_name, backend, mean_scale=mean_scale)
        init_mean_param = PPO_Pytorch._dict_to_vec(env_name, env_dist.default_parameters)
        init_stdev_param = np.array([stdev] * len(init_mean_param), dtype=np.float32)

        cem_init_mean = np.concatenate((init_mean_param, init_stdev_param))
        cem_init_stdev = np.array([1.0] * len(cem_init_mean), dtype=np.float32)

        # Make envs that will be reused for training and eval
        self.env_dist = dr.dist.Normal(env_name, backend)
        self.env_dist.backend.set_collision_detector(env_dist.root_env, collision_detector)
        self.env_dist.seed(seed)

        self.eval_envs = [gym.make('Walker2d-v2') for _ in range(num_eval_env)]

        if COMM.Get_rank() == 0:
            self.optimizer = CEMOptimizer(
                sol_dim=30,
                max_iters=300,
                popsize=self.train_params['pop_size'],
                num_elites=self.train_params['num_elites'],
                cost_function=self._cost_function,
                lower_bound=0.0,
                #TODO: setting the upper bound this way, means that
                # if the initial dimension value is 0, then the upper bound is 0
                upper_bound=cem_init_mean * 5.0,
                alpha=0.75,
                viz_dir=self.log_dir
            )

            # This is buggy
            # https://github.com/lanpa/tensorboardX/issues/345
            self.optimizer.writer.add_text('env_params', str(self.env_params), 0)
            self.optimizer.writer.add_text('train_params', str(self.train_params), 0)

            res = self.optimizer.obtain_solution(cem_init_mean, cem_init_stdev)

            path = osp.join(self.log_dir, 'res.pkl')

            with open(path, 'wb') as f:
                pickle.dump(res, f)

            COMM.Abort(0)
        else:
            while True:
                args = COMM.recv(source=0)

                r = self.train(*args)

                COMM.send(r, dest=0)

    def _cost_function(self, samples, cem_timestep):
        print(f'cem_timestep: {cem_timestep}')

        env_name = self.env_params['env_name']
        backend = self.env_params['backend']
        pop_size = self.train_params['pop_size']

        argss = [(env_name, backend,
                  self.train_params, self.env_params,
                  samples[rank][:len(samples[rank]) // 2],
                  samples[rank][len(samples[rank]) // 2:]) for rank in range(len(samples))]

        # Send args to other MPI processes
        for rank in range(1, COMM.size):
            COMM.send(argss[rank], dest=rank)

        # Obtain results for all args
        r = self.train(*argss[0])

        reses = [(0, r)]  # 0 is the rank of this process

        # Receive results from the other processes:
        for rank in range(1, COMM.size):
            r = COMM.recv(source=rank)
            reses.append((rank, r))

        reses = sorted(reses, key=lambda k: k[0])
        print(reses)

        # Get the index of the highest performing model in population
        # and write result to tensorboard
        max_idx = 0
        max_perf = max(reses[0][1])  # 0 is the result of process rank 0. 1 brings us the eval perf list

        for i, item in enumerate(reses):
            perf = max(item[1])
            if perf > max_perf:
                max_perf = perf
                max_idx = i

        # Obtain the "costs" that the CEM cost function should return
        costs = [- max(i[1]) for i in reses]
        print(costs)
        print(min(costs))
        print()

        return costs

    @classmethod
    def _dict_to_vec(cls, env_id, d):

        assert env_id == 'Walker', 'Only support Walker for in this branch'

        return np.concatenate((
            d['mass'],
            d['damping'],
            [d['gravity']]
        )).flatten().copy()

    @classmethod
    def _vec_to_dict(cls, env_id, means, stdevs):

        assert env_id == 'Walker', 'Only support Walker for in this branch'

        return dict(
            mass=means[:7],
            damping=means[7:-1],
            gravity=means[-1]
        ), dict(
            mass=stdevs[:7],
            damping=stdevs[7:-1],
            gravity=stdevs[-1]
        )
