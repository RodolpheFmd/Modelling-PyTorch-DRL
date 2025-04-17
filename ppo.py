from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

class PpoSolver:
    def __init__(self, args, nobs, nactions, run_name, device):
        super().__init__()
        self.args = args
        self.device = device
        self.nsteps = args.num_steps
        self.niterations = args.num_iterations
        self.nenvs = args.num_envs
        self.nactions = nactions
        self.lr = args.ppo_learning_rate
        self.gae = args.gae_lambda
        self.batchsize = args.ppo_batch_size
        self.minibatchsize = args.minibatch_size
        self.clip = args.clip_coef
        self.target_kl = args.target_kl
        self.ent_coef = args.ent_coef
        self.max_grad_norm = args.max_grad_norm
        self.clip_vloss = args.clip_vloss
        self.norm_adv = args.norm_adv
        self.nepochs = args.update_epochs
        self.vf_coef = args.vf_coef
        self.model_path = f"runs/{run_name}/{args.exp_name}PPO.model"
        self.policy = PpoFramework(nobs, args.ppo_hidden_layer_size, self.nactions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        self.online_memory = OnlineMemory(self.nsteps, self.nenvs, args.gamma, args.gae_lambda, nobs, self.nactions, self.device)

    @property
    def checkpoint(self):
        torch.save(self.policy.state_dict(), self.model_path)
        print(f"model saved to {self.model_path}")

    def update_lr(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.niterations
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow

    def perform_action(self, state):
        actions, logprobs, _, values = self.policy.sample_action_and_value(state)
        actions = actions.view(self.nenvs, *self.nactions)
        values = values.flatten()

        return actions, logprobs, values

    def training(self, final_state, termination):

        last_value = self.policy.sample_value(final_state).reshape(1, -1)
        batched_states, batched_actions, old_logprobs, batched_advantages, batched_returns, batched_values = self.online_memory.sample_batch(last_value, termination)

        batches = np.arange(self.batchsize)
        clipfracs = []

        for epoch in range(self.nepochs):
            np.random.shuffle(batches)
            for start in range(0, self.batchsize, self.minibatchsize):
                end = start + self.minibatchsize
                minibatches = batches[start:end]

                _, new_logprobs, entropy, new_values = self.policy.sample_action_and_value(batched_states[minibatches], batched_actions[minibatches])

                clipfracs, pi_loss, old_approx_kl, approx_kl = self.actor_loss(minibatches, batched_advantages, old_logprobs, new_logprobs, clipfracs)
                v_loss = self.critic_loss(minibatches, batched_returns, batched_values, new_values)
                entropy_loss = self.update_parameters(entropy, pi_loss, v_loss)

            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        
        return batched_values, batched_returns, v_loss.item(), pi_loss.item(), entropy_loss.item(), old_approx_kl.item(), approx_kl.item(), np.mean(clipfracs)
    
    def actor_loss(self, minibatches, batched_advantages, old_logprobs, new_logprobs, clipfracs):
        logratio = new_logprobs - old_logprobs[minibatches]
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.clip).float().mean().item()]

        minibatched_advantages = batched_advantages[minibatches]
        #if self.norm_adv:
        #    minibatched_advantages = (minibatched_advantages - minibatched_advantages.mean()) / (minibatched_advantages.std() + 1e-8)

        pg_loss1 = minibatched_advantages * ratio
        pg_loss2 = minibatched_advantages * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        pi_loss = -torch.min(pg_loss1, pg_loss2).mean()

        return clipfracs, pi_loss, old_approx_kl, approx_kl
    
    def critic_loss(self, minibatches, batched_returns, batched_values, sampled_values):
        newvalue = sampled_values.view(-1)
        v_loss_unclipped = (newvalue - batched_returns[minibatches]) ** 2
        #if self.clip_vloss:
            #v_loss_unclipped = (newvalue - batched_returns[minibatches]) ** 2
            #v_clipped = batched_values[minibatches] + torch.clamp(newvalue - batched_values[minibatches], -self.clip, self.clip)
            #v_loss_clipped = (v_clipped - batched_returns[minibatches]) ** 2
            #v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            #v_loss = 0.5 * v_loss_max.mean()
        #else:
            #v_loss = 0.5 * ((newvalue - batched_returns[minibatches]) ** 2).mean()
        return v_loss_unclipped.mean()
    
    def update_parameters(self, entropy, pi_loss, v_loss):
        entropy_loss = -entropy.mean()
        loss = pi_loss + self.ent_coef * entropy_loss + v_loss * self.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return entropy_loss

class PpoFramework(nn.Module):
    def __init__(self, nobs, hidden_layer_size, nactions):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(np.array(nobs).prod(), hidden_layer_size)),
                                    nn.ReLU(),
                                    layer_init(nn.Linear(hidden_layer_size, hidden_layer_size)),
                                    nn.ReLU(),
                                    layer_init(nn.Linear(hidden_layer_size, 1), std=1.0))
        self.actor = nn.Sequential(layer_init(nn.Linear(np.array(nobs).prod(), hidden_layer_size)),
                                   nn.ReLU(),
                                   layer_init(nn.Linear(hidden_layer_size, hidden_layer_size)),
                                   nn.ReLU(),
                                   layer_init(nn.Linear(hidden_layer_size, np.prod(nactions)), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(nactions)))

    def sample_value(self, state):
        return self.critic(state)
    
    def forward(self, state):
        actor_logits = self.actor(state)
        values = self.critic(state)

        return actor_logits, values

    def sample_action_and_value(self, state, action=None):
        actor_logits, values = self.forward(state)
        action_logstd = self.actor_logstd.expand_as(actor_logits)
        action_std = torch.exp(action_logstd)
        probs = Normal(actor_logits, action_std)

        action = action.view(actor_logits.size()) if action is not None else probs.sample()
        logprobs = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        
        return action, logprobs, entropy, values

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class OnlineMemory:
    def __init__(self, nsteps, nenvs, gamma, gae_lambda, nobs, nactions, device):
        super().__init__()

        self.nsteps = nsteps
        self.nenvs = nenvs
        self.device = device
        self.gamma = gamma
        self.gae = gae_lambda
        self.nobs = nobs
        self.nactions = nactions

        self.reset()

    def reset(self):
        self.states = torch.zeros((self.nsteps, self.nenvs) + self.nobs).to(self.device)
        self.actions = torch.zeros((self.nsteps, self.nenvs) + self.nactions).to(self.device)
        self.logprobs = torch.zeros((self.nsteps, self.nenvs)).to(self.device)
        self.rewards = torch.zeros((self.nsteps, self.nenvs)).to(self.device)
        self.dones = torch.zeros((self.nsteps, self.nenvs)).to(self.device)
        self.values = torch.zeros((self.nsteps, self.nenvs)).to(self.device)

    def sample_batch(self, final_state, termination):
        returns = self.advantage_return(final_state, termination)
        b_obs = self.states.reshape((-1,) + self.nobs)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.nactions)
        b_advantages = self.advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values
    
    def advantage_return(self, next_value, termination):
        print(f'PPO Reward: {self.rewards.mean()}')
        with torch.no_grad():
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - termination
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + self.gamma * self.gae * nextnonterminal * lastgaelam
            return self.advantages + self.values
        
    def save_trajectories(self, step, states, actions, logprobs, rewards, values, dones):
        self.states[step] = states
        self.values[step] = values
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones