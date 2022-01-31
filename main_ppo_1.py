import gym
import torch  # Torch version :1.9.0+cpu
from ppo_agent_1 import Agent
from network_1 import ActorCriticNetwork
from ppo_agent_1 import gif, plot
from PIL import Image
import random
import pickle

torch.manual_seed(14)

# Different env to test my PPO
pendulum = 'Pendulum-v0'
lunar = 'LunarLanderContinuous-v2'
biped = 'BipedalWalker-v3'

TESTING = False         # Set testing and test the model with gif function to record
TRAINING_OLD = False     # Use old model to re-start training
RECORD = False
ENV_NAME = biped        # What env to use
name = './Models_ppo/PPO_' + str(ENV_NAME) + '_actor' + '.pth'

# if training
TRAINING_STEPS = 100_000_000

if TESTING:
    env = gym.make(ENV_NAME)
    n_acts = env.action_space.shape[0]
    n_obs = env.observation_space.shape[0]

    model = ActorCriticNetwork(n_obs, n_acts)
    model.load_state_dict(torch.load(name))

    TEST_PPO = 5
    max_steps = 1000
    for i in range(TEST_PPO):
        obs = env.reset()
        rewards = 0
        done = False
        images = []
        for step in range(max_steps):
            if i % 2 == 0 and RECORD:
                # Render to frames buffer
                image = (env.render(mode="rgb_array"))
                image = Image.fromarray(image)
                images.append(image)

            env.render()
            obs = torch.tensor(obs, dtype=torch.float)
            act = model(obs).detach().numpy()
            obs_, rew, done, _ = env.step(act)
            rewards += rew
            if done:
                break
            step += 1
            obs = obs_

        print(f'Episode: {i} | Rewards: {rewards} | Steps taken: {step}')

        if i % 2 == 0 and RECORD:  # Record
            num = random.randint(0, 100000)
            gif(images, 'gif_ppo_mod_' + 'intermediate' + str(num) + 'trained.gif')

    env.close()

else:
    New_Agent = Agent(ENV_NAME, TRAINING_OLD)
    New_Agent.learn(TRAINING_STEPS)

    # plot
    with open('./Models_ppo/PPO_' + str(ENV_NAME) +'_previous_mod.pkl', 'rb') as file:
        reward_list = pickle.load(file)
        actor_loss_t = pickle.load(file)
        critic_loss_t = pickle.load(file)

    plot(reward_list, actor_loss_t, critic_loss_t, ENV_NAME)



