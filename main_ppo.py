import gym
import torch  # Torch version :1.9.0+cpu
from ppo_1 import Agent
from ppo_batch import AgentBatch
from network import ActorCriticNetwork
from utils import gif, plot
from PIL import Image
import random
import pickle

if __name__ == '__main__':

    torch.manual_seed(14)

    # Different env to test my PPO
    pendulum = 'Pendulum-v0'
    lunar = 'LunarLanderContinuous-v2'
    biped = 'BipedalWalker-v3'
    mountain_car = 'MountainCarContinuous-v0'

    BATCH_PPO = True            # which model to use

    TESTING = False             # Set testing and test the model with gif function to record
    RECORD = False              # record model if testing
    TRAINING_OLD = False        # Use old model to re-start training

    ENV_NAME = lunar         # What env to use

    if BATCH_PPO:
        name = './Models_ppo/PPO_batch_' + str(ENV_NAME) + '_actor' + '.pth'
    else:
        name = './Models_ppo/PPO_' + str(ENV_NAME) + '_actor' + '.pth'

    # if training
    TRAINING_STEPS = 100_000_000

    if TESTING:
        env = gym.make(ENV_NAME)
        n_acts = env.action_space.shape[0]
        n_obs = env.observation_space.shape[0]

        print('using model:', name)
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
                gif(images, 'gif_ppo_mod_' + ENV_NAME + "_" + str(num) + 'trained.gif')

        env.close()

    else:
        if BATCH_PPO:
            new_agent = AgentBatch(ENV_NAME, TRAINING_OLD)
            new_agent.learn(TRAINING_STEPS)
        else:
            new_agent = Agent(ENV_NAME, TRAINING_OLD)
            new_agent.learn(TRAINING_STEPS)

        # plot
        if BATCH_PPO:
            name_ = './Models_ppo/PPO_batch_' + str(ENV_NAME) + '_previous_mod.pkl'
        else:
            name_ = './Models_ppo/PPO_' + str(ENV_NAME) + '_previous_mod.pkl'

        with open(name_, 'rb') as file:
            reward_list = pickle.load(file)
            actor_loss_t = pickle.load(file)
            critic_loss_t = pickle.load(file)
            losses_t = pickle.load(file)

        plot(reward_list, actor_loss_t, critic_loss_t, losses_t, ENV_NAME)



