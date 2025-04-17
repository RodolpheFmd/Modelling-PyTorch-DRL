from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

class SacSolver:
    def __init__(self, args, nobs, nactions, run_name, device):
        super().__init__()
        self.args = args
        self.device = device
        self.nsteps = args.num_steps
        self.niterations = args.num_iterations
        self.nenvs = args.num_envs
        self.nactions = nactions
        self.target_entropy = -np.prod(self.nactions)
        self.lr = args.sac_learning_rate
        self.nepochs = args.update_epochs
        self.gamma = args.gamma
        self.max_grad_norm = args.max_grad_norm
        self.nepochs = args.update_epochs
        self.actor_model_path = f"runs/{run_name}/{args.exp_name}_Actor_SAC.model"
        self.critic_model_path = f"runs/{run_name}/{args.exp_name}_Critic_SAC.model"

        self.device = device
        self.run_name = run_name

        self.actor = Actor(nobs, args.sac_hidden_layer_size, self.nactions,  args.log_sigmin, args.log_sigmax, args.action_scale, args.action_bias, self.lr)
        self.critic = Critic(nobs, args.sac_hidden_layer_size, self.nactions, args.tau, self.lr)

        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(args.alpha, dtype=torch.float32)))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr, eps=1e-5)

        self.offline_memory = ReplayBuffer(args.memory_capacity, self.nenvs, nobs, self.nactions, args.sac_batch_size, args.reward_signal, self.device)
        self.action_log = []

    @property
    def checkpoint(self):
        torch.save(self.actor.state_dict(), self.actor_model_path)
        torch.save(self.critic.state_dict(), self.critic_model_path)
    
    def perform_action(self, state):
        actions = self.actor.sample_actions(state)[0]
        self.action_log.append(actions.detach().cpu())
        actions = actions.view(self.nenvs, *self.nactions)
        return actions

    @property
    def reset_learning_data(self):
        self.rewards = torch.zeros((self.nepochs), dtype=torch.float32, requires_grad=False, device=self.device)
        self.pi_loss = torch.zeros((self.nepochs), dtype=torch.float32, requires_grad=False, device=self.device)
        self.q1_loss = torch.zeros((self.nepochs), dtype=torch.float32, requires_grad=False, device=self.device)
        self.q2_loss = torch.zeros((self.nepochs), dtype=torch.float32, requires_grad=False, device=self.device)
        self.alpha_loss = torch.zeros((self.nepochs), dtype=torch.float32, requires_grad=False, device=self.device)

    def analysis_action(self):
        all_actions = torch.cat(self.action_log).view(-1)
        bin_edges = torch.arange(-1.1, 1.1, 0.1)
        hist, _ = torch.histogram(all_actions, bins=bin_edges)
        percentages = hist.float() / hist.sum() * 100

        for i in range(len(hist)):
            print(f"Bin {bin_edges[i]:.1f} to {bin_edges[i+1]:.1f}: {percentages[i]:.2f}%")

    def training(self):
        #self.analysis_action()
        self.reset_learning_data
        for epoch in range(self.nepochs):
            states, actions, rewards, next_states, dones = self.offline_memory.sample()
            self.q1_loss[epoch], self.q2_loss[epoch] = self.update_critic(states, actions, next_states, rewards, dones)

            #q1_loss, q2_loss = self.update_critic(states, actions, next_states, rewards, dones)
            self.pi_loss[epoch], log_probs = self.update_actor(states)
            self.rewards[epoch] = rewards.mean()
            self.update_alpha(log_probs)
            
            self.critic.soft_update()
        print(f'SAC Reward: {self.rewards.mean()}')
        
        return self.q1_loss.mean(), self.q2_loss.mean(), self.pi_loss.mean(), self.alpha

    def target_q_values(self, next_states, rewards, dones):
        next_actions, next_log_probs = self.actor.sample_actions(next_states)
        next_q1, next_q2 = self.critic.sample_actionstate_target_values(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2) 

        target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)

        return target_q

    def critic_loss(self, states, actions, next_states, rewards, dones):
        q1_values, q2_values = self.critic.sample_actionstate_values(states, actions)
        with torch.no_grad():
            target_q = self.target_q_values(next_states, rewards, dones)

        q1_loss = F.mse_loss(q1_values, target_q)
        q2_loss = F.mse_loss(q2_values, target_q)

        return q1_loss, q2_loss
    
    def update_critic(self, states, actions, next_states, rewards, dones):
        
        q1_loss, q2_loss = self.critic_loss(states, actions, next_states, rewards, dones)

        self.critic.q1.optimizer.zero_grad()
        q1_loss.backward()
        self.critic.q1.optimizer.step()
        self.critic.q2.optimizer.zero_grad()
        q2_loss.backward()
        self.critic.q2.optimizer.step()
        return q1_loss, q2_loss
    
    def actor_loss(self, states):
        noised_actions, log_probs = self.actor.sample_actions(states, reparametrization=True)

        with torch.no_grad():
            q1, q2 = self.critic.sample_actionstate_values(states, noised_actions)
        min_q = torch.min(q1, q2)
        loss = torch.mean(self.alpha * log_probs - min_q)

        return loss, log_probs
    
    def update_actor(self, states):
        actor_loss, log_probs = self.actor_loss(states)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        return actor_loss, log_probs
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def update_alpha(self, log_probs):
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

class Actor(nn.Module):
    def __init__(self, nobs, hidden_layer_size, nactions, log_sigmin, log_sigmax, action_scale, action_bias, lr):
        super().__init__()

        self.log_sigmin = log_sigmin
        self.log_sigmax = log_sigmax

        self.action_scale = action_scale
        self.action_bias = action_bias

        self.model = nn.Sequential(nn.Linear(np.array(nobs).prod(), hidden_layer_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_size, hidden_layer_size),
                                   nn.ReLU())
        self.mu_head = nn.Linear(hidden_layer_size, np.prod(nactions))
        self.log_std_head = nn.Linear(hidden_layer_size, np.prod(nactions))

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def forward(self, state):
        logits = self.model(state)
        mean, log_std = self.mu_head(logits), self.log_std_head(logits)
        log_std = torch.clamp(log_std, min=self.log_sigmin, max=self.log_sigmax)
        sigma = log_std.exp()

        return mean, sigma
    
    def sample_actions(self, state, reparametrization=False):
        mean, sigma = self.forward(state)
        dist = Normal(mean, sigma)

        unsquashed_actions = dist.sample() if not reparametrization else dist.rsample()
        squashed_actions = torch.tanh(unsquashed_actions)
        action = squashed_actions * self.action_scale + self.action_bias

        log_prob = dist.log_prob(unsquashed_actions)
        
        log_prob -= 2 * (np.log(2) - squashed_actions - F.softplus(-2 * squashed_actions)) 
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob

class Critic(nn.Module):
    def __init__(self, nobs, hidden_layer_size, nactions, tau, lr):
        super().__init__()

        self.q1 = QNetwork(nobs, hidden_layer_size, nactions, lr)
        self.q2 = QNetwork(nobs, hidden_layer_size, nactions, lr)

        self.q1_target = QNetwork(nobs, hidden_layer_size, nactions, lr)
        self.q2_target = QNetwork(nobs, hidden_layer_size, nactions, lr)
        self.tau = tau

        self.hard_update()
    
    def sample_actionstate_values(self, state, action):
        action_state = torch.cat([state, (1+action)/2], dim=-1)
        q1_value = self.q1.forward(action_state)
        q2_value = self.q2.forward(action_state)
        #print("Q1 mean:", q1_value.mean().item(), "Q2 mean:", q2_value.mean().item())
        return q1_value.squeeze(), q2_value.squeeze()
    
    def sample_actionstate_target_values(self, state, action):
        action_state = torch.cat([state, (1+action)/2], dim=-1)
        q1_target_value = self.q1_target.forward(action_state)
        q2_target_value = self.q2_target.forward(action_state)
        #print("Q1 target mean:", q1_target_value.mean().item(), "Q2 target mean:", q2_target_value.mean().item())
        return q1_target_value.squeeze(), q2_target_value.squeeze()
    
    def hard_update(self):
        for source, target in zip([self.q1, self.q2], [self.q1_target, self.q2_target]):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(param.data)
    
    def soft_update(self):
        for source, target in zip([self.q1, self.q2], [self.q1_target, self.q2_target]):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

class QNetwork(nn.Module):
    def __init__(self, nobs, hidden_layer_size, nactions, lr):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(np.array(nobs).prod() + np.prod(nactions), hidden_layer_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_size, hidden_layer_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_size, 1))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
    
    def forward(self, action_state):
        q_value = self.model(action_state)
        return q_value
    
class ReplayBuffer:
    def __init__(self, memory_cap, nenvs, nobs, nactions, batchsize, reward_signal, device):
        super().__init__()
        self.memory_cap = memory_cap
        self.nenvs = nenvs
        self.nobs = nobs
        self.nactions = nactions
        self.batchsize = batchsize
        self.device = device
        self.reward_signal = reward_signal

        self.size = 0
        self.idx = 0

        self.reset()
    
    def reset(self):
        self.states = torch.zeros((self.memory_cap,) + self.nobs).to(self.device)
        self.actions = torch.zeros((self.memory_cap,) + self.nactions).to(self.device)
        self.rewards = torch.zeros((self.memory_cap,)).to(self.device)
        self.next_states = torch.zeros((self.memory_cap,) + self.nobs).to(self.device)
        self.dones = torch.zeros((self.memory_cap,)).to(self.device)

    def save_trajectories(self, state, action, reward, next_state, done):
        
        batch_size = state.shape[0]
        end_idx = self.idx + batch_size
        if end_idx <= self.memory_cap:
            self.states[self.idx:end_idx] = state
            self.actions[self.idx:end_idx] = action
            self.rewards[self.idx:end_idx] = reward*self.reward_signal
            self.next_states[self.idx:end_idx] = next_state
            self.dones[self.idx:end_idx] = done
        else:
            first_part = self.memory_cap - self.idx
            second_part = batch_size - first_part

            self.states[self.idx:] = state[:first_part]
            self.actions[self.idx:] = action[:first_part]
            self.rewards[self.idx:] = reward[:first_part]*self.reward_signal
            self.next_states[self.idx:] = next_state[:first_part]
            self.dones[self.idx:] = done[:first_part]

            self.states[:second_part] = state[first_part:]
            self.actions[:second_part] = action[first_part:]
            self.rewards[:second_part] = reward[first_part:]*self.reward_signal
            self.next_states[:second_part] = next_state[first_part:]
            self.dones[:second_part] = done[first_part:]

        self.idx = (self.idx + batch_size) % self.memory_cap
        self.size = min(self.size + batch_size, self.memory_cap)

    def sample(self):
        idx = torch.randint(0, self.size, (self.batchsize,), device=self.device)
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.next_states[idx]
        done = self.dones[idx]

        return state, action.reshape(*action.shape[:-2], -1), reward, next_state, done