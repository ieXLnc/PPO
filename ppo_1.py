import gym
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

import torch  # Torch version :1.9.0+cpu
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network_1 import ActorCriticNetwork

torch.manual_seed(14)
np.random.default_rng(14)


class Agent:
    def __init__(self, name, training_old):
        self._init_hyperparameters()

        self.env = gym.make(name)
        self.n_actions = self.env.action_space.shape[0]
        self.n_observations = self.env.observation_space.shape[0]

        # Create networks
        self.actor = ActorCriticNetwork(self.n_observations, self.n_actions)
        self.critic = ActorCriticNetwork(self.n_observations, 1)
        self.name = './Models_ppo/PPO_' + str(name)
        self.actor_name = self.name + '_actor' + '.pth'
        self.critic_name = self.name + '_critic' + '.pth'
        self.drop_name_params = self.name + '_previous_mod.pkl'

        if training_old:
            self.actor.load_state_dict(torch.load(self.actor_name))
            self.critic.load_state_dict(torch.load(self.critic_name))
            print('... Load Actor/Critic ppo_1 from previous training ...')
        else:
            print('New Actor/Critique models created from ppo_1...')

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)  # Creates a tensor of size filled with values
        self.cov_mat = torch.diag(self.cov_var)

        self.log = {
            'timesteps': 0,
            'learning iterations': 0,
            'batch_lens': [],
            'rewards': [],
            'actor_loss': [],
            'critic_loss': [],
            'previous_mean': [],
            'actor_loss_tracks': [],
            'critic_loss_tracks': [],
            'losses': [],
            'losses_tracks': []
        }
        # init log if previous model
        if training_old:
            if os.path.exists(self.drop_name_params):
                with open(self.drop_name_params, 'rb') as file:
                    previous_means = pickle.load(file)
                    actor_loss_t = pickle.load(file)
                    critic_loss_t = pickle.load(file)
                    losses_t = pickle.load(file)
                self.log['previous_mean'] = previous_means
                self.log['actor_loss_tracks'] = actor_loss_t
                self.log['critic_loss_tracks'] = critic_loss_t
                self.log['losses_tracks'] = losses_t
                print('previous means loaded...')
                print(f'best previous model: {self.log["previous_mean"][np.argmax(self.log["previous_mean"])]}')
            else:
                print('No previous means detected.')

    def _init_hyperparameters(self):
        self.gamma = 0.95                   # gamma
        self.update_model = 5               # How many times we update the model
        self.clip = 0.2                     # clip params
        self.lr = 0.0005                     # learning rate
        self.exploration_steps = 4096       # number of steps we take to learn
        self. mini_batch_size = 64
        # self.max_steps_per_ep = 1600        # number of steps by episode
        self.value_early_stop = -150         # implemented an early stop
        self.early_stop = False

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        # prediction of our actor
        mean = self.actor(state)
        # value = self.critic(state)
        # That we have to transform in a distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()  # get a random sample
        log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action.detach().numpy(), log_probs.detach(), entropy  # value.detach()

    def evaluate(self, states, actions):
        # get the new values from critic
        values = self.critic(states).squeeze()
        # get the predictions from actor updated
        mean = self.actor(states)
        dist = MultivariateNormal(mean, self.cov_mat)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return new_log_probs, values, entropy

    def compute_discounted_rewards(self, batch_rewards, batch_dones):
        batch_discounted_reward = []
        discounted = 0
        for t in reversed(range(len(batch_rewards))):
            if batch_dones[t]:
                discounted = 0

            discounted = batch_rewards[t] + discounted * self.gamma
            batch_discounted_reward.insert(0, discounted)

        return batch_discounted_reward

    def explore_env(self):
        # initialize all lists
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        batch_discounted_reward = []
        batch_lens = []
        entropy = 0

        step = 0
        current_state = self.env.reset()
        while step < self.exploration_steps:

            action, log_prob, entropy = self.get_action(current_state)
            new_state, reward, done, _ = self.env.step(action)

            batch_states.append(current_state)        # Array 1D with 3 values
            batch_actions.append(action)              # 1D arr with 1 val
            batch_log_probs.append(log_prob)          # tensor obj with one val
            batch_rewards.append(reward)              # Array 1D
            batch_dones.append(done)                  # Bool
            # add reward
            step += 1                                 # steps current explorations
            if done:
                current_state = self.env.reset()
                batch_lens.append(step+1)

            current_state = new_state

        batch_discounted_reward = self.compute_discounted_rewards(batch_rewards, batch_dones)

        self.log['rewards'] = batch_rewards
        self.log['batch_lens'] = batch_lens

        return torch.tensor(batch_states, dtype=torch.float), \
               torch.tensor(batch_actions, dtype=torch.float), \
               torch.tensor(batch_log_probs, dtype=torch.float), \
               torch.tensor(batch_discounted_reward, dtype=torch.float), entropy

    def test_model(self):
        obs = self.env.reset()
        d = False
        reward_ep = 0
        while not d:
            obs = torch.tensor(obs, dtype=torch.float)
            action = self.actor(obs).detach().numpy()
            obs_, rew, d, _ = self.env.step(action)
            reward_ep += rew
            obs = obs_
        return reward_ep

    def learn(self, learning_steps):
        step = 0
        i_learn = 0
        while step < learning_steps:
            # load all trajectories
            batch_states, batch_actions, batch_log_probs, batch_discounted_reward, entropy = self.explore_env()

            step += self.exploration_steps
            i_learn += 1

            _, val, _ = self.evaluate(batch_states, batch_actions)
            advantage = batch_discounted_reward - val.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.update_model):

                batch_new_probs, values, entropy = self.evaluate(batch_states, batch_actions)

                ratios = torch.exp(batch_new_probs - batch_log_probs)
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage

                actor_loss = (- torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(values, batch_discounted_reward)
                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                self.log['actor_loss'].append(actor_loss.detach())
                self.log['critic_loss'].append(critic_loss.detach())
                self.log['losses'].append(loss.detach())

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()

            self.log['timesteps'] = step
            self.log['learning iterations'] = i_learn

            reward_ep = np.mean([self.test_model() for _ in range(10)])
            self.log['rewards'] = reward_ep

            if self.early_stop:
                print('stop')
                break

            self.summary()

    def summary(self):
        timesteps = self.log['timesteps']
        i_learn = self.log['learning iterations']

        mean_rew_ep = self.log['rewards']
        mean_actor_loss = np.mean([losses.float().mean() for losses in self.log['actor_loss']])
        mean_critic_loss = np.mean([losses.float().mean() for losses in self.log['critic_loss']])
        mean_loss = np.mean([losses.float().mean() for losses in self.log['losses']])

        print(f"-------------- Iteration so far #{i_learn} --------------")
        print(f"Average Episodic Lenght: {self.exploration_steps}")
        print(f"Average Episodic Rewards: {mean_rew_ep}")

        if len(self.log['previous_mean']) > 0:
            if mean_rew_ep > self.log["previous_mean"][np.argmax(self.log["previous_mean"])]:
                torch.save(self.actor.state_dict(), self.actor_name)
                torch.save(self.critic.state_dict(), self.critic_name)
                self.log['previous_mean'].append(mean_rew_ep)
                self.log['actor_loss_tracks'].append(mean_actor_loss)
                self.log['critic_loss_tracks'].append(mean_critic_loss)
                self.log['losses_tracks'].append(mean_loss)
                with open(self.drop_name_params, 'wb') as handle:
                    pickle.dump(self.log['previous_mean'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(self.log['actor_loss_tracks'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(self.log['critic_loss_tracks'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(self.log['losses_tracks'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'New Model saved with a reward mean of {mean_rew_ep}')
            else:
                print(f"Best model so far: {self.log['previous_mean'][np.argmax(self.log['previous_mean'])]}")

        print(f"Average Actor Loss: {mean_actor_loss}")
        print(f"Average Critic Loss: {mean_critic_loss}")
        print(f"Average Losses: {mean_loss}")
        print(f"Timesteps so Far {timesteps}")

        if mean_rew_ep > self.value_early_stop:
            self.early_stop = True
            print(f'Model trained with {mean_rew_ep}: early stop activated')
            torch.save(self.actor.state_dict(), self.actor_name)
            torch.save(self.critic.state_dict(), self.critic_name)
            self.log['previous_mean'].append(mean_rew_ep)
            self.log['actor_loss_tracks'].append(mean_actor_loss)
            self.log['critic_loss_tracks'].append(mean_critic_loss)
            self.log['losses_tracks'].append(mean_loss)
            with open(self.drop_name_params, 'wb') as handle:
                pickle.dump(self.log['previous_mean'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.log['actor_loss_tracks'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.log['critic_loss_tracks'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.log['losses_tracks'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Models saved.')

        print(f"---------------------------------------------------------")

        self.log['previous_mean'].append(mean_rew_ep)
        self.log['actor_loss_tracks'].append(mean_actor_loss)
        self.log['critic_loss_tracks'].append(mean_critic_loss)
        self.log['losses_tracks'].append(mean_loss)

        self.log['rewards'] = []
        self.log['actor_losses'] = []
        self.log['critic_loss'] = []
        self.log['losses'] = []