from models import Policy
from models import Value

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

class Agent:
    def __init__(self, args):
        self.name = args['name']
        self.device = args['device']
        self.save_dir = args['save_dir']
        self.checkpoint_dir=f'{self.save_dir}/checkpoint'
        self.discount_factor = args['discount_factor']
        self.value_lr = args['value_lr']
        self.policy_lr = args['policy_lr']
        self.value_epochs = args['value_epochs']
        self.policy_epochs = args['policy_epochs']
        self.clip_value = args['clip_value']
        self.gae_coeff = args['gae_coeff']
        self.ent_coeff = args['ent_coeff']
        self.num_envs = args['num_envs']
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.action_bound_min = torch.tensor(args['action_bound_min'], device=self.device)
        self.action_bound_max = torch.tensor(args['action_bound_max'], device=self.device)
        self.max_kl = args['max_kl']
        self.max_grad_norm = args['max_grad_norm']

        self.policy = Policy(args).to(self.device)
        self.value = Value(args).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.load()


    def normalizeAction(self, a:torch.Tensor):
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a:torch.Tensor):
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def getAction(self, state, is_train):
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self.unnormalizeAction(mean + noise*std)
        else:
            action = self.unnormalizeAction(mean)
        return action

    def getGaesTargets(self, rewards, values, dones, fails, next_values):
        rewards = rewards.reshape((-1, self.num_envs))
        values = values.reshape((-1, self.num_envs))
        dones = dones.reshape((-1, self.num_envs))
        fails = fails.reshape((-1, self.num_envs))
        next_values = next_values.reshape((-1, self.num_envs))
        deltas = rewards + (1.0 - fails)*self.discount_factor*next_values - values
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t])*self.discount_factor*self.gae_coeff*gaes[t + 1]
        targets = values + gaes
        return gaes.reshape(-1), targets.reshape(-1)

    def getEntropy(self, states:torch.Tensor):
        '''
        return scalar tensor for entropy value.
        input:
            states:     Tensor(n_steps, state_dim)
        output:
            entropy:    Tensor(,)
        '''
        means, log_stds, stds = self.policy(states)
        normal = torch.distributions.Normal(means, stds)
        entropy = torch.mean(torch.sum(normal.entropy(), dim=1))
        return entropy

    def train(self, states_tensor, actions_tensor, rewards_tensor, dones_tensor, fails_tensor, next_states_tensor):
        # convert to numpy array
        rewards = rewards_tensor.detach().cpu().numpy()
        dones = dones_tensor.detach().cpu().numpy()
        fails = fails_tensor.detach().cpu().numpy()

        # convert to tensor
        norm_actions_tensor = self.normalizeAction(actions_tensor)

        # get GAEs and Tagets
        values_tensor = self.value(states_tensor)
        next_values_tensor = self.value(next_states_tensor)
        values = values_tensor.detach().cpu().numpy()
        next_values = next_values_tensor.detach().cpu().numpy()
        gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)
        gaes = (gaes - np.mean(gaes))/(np.std(gaes) + EPS)
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float)

        # ========== for policy update ========== #
        means, log_stds, stds = self.policy(states_tensor)
        old_means = means.detach().clone()
        old_stds = stds.detach().clone()
        old_dist = torch.distributions.Normal(old_means, old_stds)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions_tensor), dim=1)
        for _ in range(self.policy_epochs):
            means, log_stds, stds = self.policy(states_tensor)
            entropy = self.getEntropy(states_tensor)
            dist = torch.distributions.Normal(means, stds)
            log_probs = torch.sum(dist.log_prob(norm_actions_tensor), dim=1)
            ratios = torch.exp(log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, min=1.0-self.clip_value, max=1.0+self.clip_value)
            policy_loss = -(torch.mean(torch.minimum(gaes_tensor*ratios, gaes_tensor*clipped_ratios)) + self.ent_coeff*entropy)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            kl = self.getKL(states_tensor, old_means, old_stds)
            if self.max_kl < kl: break
        # ======================================= #

        # =========== for value update =========== #
        for _ in range(self.value_epochs):
            value_loss = torch.mean(torch.square(self.value(states_tensor) - targets_tensor))
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
        # ======================================== #

        return value_loss.item(), policy_loss.item(), kl.item(), entropy.item()

    def getKL(self, states, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl

    def save(self):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'p_optim': self.policy_optimizer.state_dict(),
            'v_optim': self.value_optimizer.state_dict(),
            }, f"{self.checkpoint_dir}/model.pt")
        print(f'[{self.name}] save success.')

    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.policy_optimizer.load_state_dict(checkpoint['p_optim'])
            self.value_optimizer.load_state_dict(checkpoint['v_optim'])
            print(f'[{self.name}] load success.')
        else:
            self.policy.initialize()
            self.value.initialize()
            print(f'[{self.name}] load fail.')
