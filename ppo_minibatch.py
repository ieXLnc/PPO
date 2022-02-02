import gym
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network_1 import ActorCriticNetwork
import matplotlib.pyplot as plt
from IPython.display import clear_output


# create env
env = gym.make('Pendulum-v0')
n_acts = env.action_space.shape[0]
n_obs = env.observation_space.shape[0]

# create actor/critique
Actor = ActorCriticNetwork(n_obs, n_acts)
Critic = ActorCriticNetwork(n_obs, 1)

lr = 3e-4
actor_optimizer = Adam(Actor.parameters(), lr=lr)
critic_optimizer = Adam(Critic.parameters(), lr=lr)

cov_var = torch.full(size=(n_acts,), fill_value=0.5)  # Creates a tensor of size filled with values
cov_mat = torch.diag(cov_var)


def get_action(state):
    state = torch.tensor(state, dtype=torch.float)  # state to torch
    # get from the models
    mean = Actor(state)
    value = Critic(state)
    dist = MultivariateNormal(mean, cov_mat)
    action = dist.sample()  # get a random sample
    entropy = dist.entropy().mean()
    log_probs = dist.log_prob(action)

    return action.detach().numpy(), log_probs, entropy, value.detach()


def evaluate(states, actions):
    mean = Actor(states)
    value = Critic(states)
    dist = MultivariateNormal(mean, cov_mat)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    return new_log_probs, value, entropy


def get_gae(rewards, values, last_value, dones, lmbda=0.95, gamma=0.99):
    values = values + [last_value]
    gae = 0
    returns = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * dones[t] - values[t]
        gae = delta + gamma * lmbda * dones[t] * gae
        returns.insert(0, gae + values[t])
    return returns


def get_mini_batches(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :],\
              advantage[rand_ids, :],


def ppo_update(n_updates, mini_batch_size, states, actions, log_probs, returns, advantage, clip_param=0.2):
    for _ in range(n_updates):
        for states_, actions_, old_probs, returns_, advantage_ in get_mini_batches(mini_batch_size, states,
                                                                               actions, log_probs, returns, advantage):

            new_log_probs, value, entropy = evaluate(states_, actions_)

            ratios = (new_log_probs - old_probs).exp()
            surr1 = ratios * advantage_
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantage_

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (returns_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()


def test_model(vis=False):
    obs = env.reset()
    d = False
    total_reward = 0
    while not d:
        if vis:
            env.render()
        obs = torch.tensor(obs, dtype=torch.float)
        act = Actor(obs).detach().numpy()
        obs_, rew, d, _ = env.step(act)
        if d:
            break
        obs = obs_
        total_reward += rew

    return total_reward


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


# Create main function and code the other when needed

# part to initialize all params
MAX_FRAMES = 15000
STEP_BATCH = 20
n_updates = 4
mini_batch_size = 5
test_rewards = []

# function
state = env.reset()
frame_idx = 0
while frame_idx < MAX_FRAMES:
    # init all the lists
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    entropy = 0

    for _ in range(STEP_BATCH):

        # get action (np), log_prob (ten), entropy (mean), value (np)
        action, log_prob, entropy, value = get_action(state)
        new_state, reward, done, _ = env.step(action)           # get new info for the step taken
        # record all
        states.append(state)            # list of array with 3 obs
        actions.append(action)          # list of array
        log_probs.append(log_prob)      # list of tensor with graph
        values.append(value)            # list of tensor without graph
        rewards.append(reward)          # list of array
        dones.append(done)              # list of array
        entropy += entropy

        state = new_state
        frame_idx += 1
        # test function
        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_model() for _ in range(10)])
            test_rewards.append(test_reward)
            print('frame_idx', frame_idx)
            print('test_rewards', test_rewards[-1])
            #plot(frame_idx, test_rewards)

    # dont append the next value because only use to calculate GAE here
    _, _, _, last_value = get_action(new_state)  # get last value to calculate GAE

    returns = get_gae(rewards, values, last_value, dones)

    returns = torch.tensor(returns, dtype=torch.float)
    values = torch.tensor(values, dtype=torch.float)
    advantage = returns - values
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)  # normalize advantages

    # every used variables in tensor and shape [20,1]
    returns = returns.unsqueeze(1)  # unsqueeze reshape tensor to have [20, 1]
    values = values.unsqueeze(1)
    advantage = advantage.unsqueeze(1)

    states = torch.tensor(states, dtype=torch.float)  # shape [20,3]
    actions = torch.tensor(actions, dtype=torch.float)
    log_probs = torch.tensor(log_probs, dtype=torch.float).unsqueeze(1)

    ppo_update(n_updates, mini_batch_size, states, actions, log_probs, returns, advantage)

