# PPO


## PPO_1:

Implementation of two Proximal Policy Optimization agents to solve severals continuous environment in the openai gym space.
As I continue to try to learn reinforcement learning, I stopped by the proximal policy optimation algorithm to learn about on-policy models and the Actor/Critic method.

Many articles, Github repo, posts or videos were particularly useful to learn about PPO, such as [this 4 part tutorial on Medium](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) or [this video](https://www.youtube.com/watch?v=5P7I-xPq8u8) on PPO.

The first model I implemented (PPO_1) takes the following hyperparams:

- Horizon (number of steps the agent takes to update itself at each learning iteration): 4800
- Update_epoch (number of epochs of update of the model): 5
- Learning rate: 0.003
- Gamma : 0.99
- Clipping range: 0.2
- Entropy Coefficient: 0.01
- Early stop (depends on the environment).

The Actor and Critic come from the same Actor/Critic structure in network.py. They are composed of two hidden layers with 256 nodes each, taking the observation_space as input and action_space as output (Actor) or output = 1 (Critic). I initialize the weights of the model with the normal_ function from Pytorch to fill the input tensor with normally distributed values.

The agent learns by taking a fixed set of steps (his horizon), calculating the discounted rewards and the advantages with (discounted_rewards - values) and normalizing it. He then updates (x Update_epoch) itself comparing his old policy (actor prediction while exploring) to the new policy (new prediction of the actor for n_update) and calculating the new value (new prediction of the critic for the actions). Both actor and critic losses are calculated and create the general loss with entropy to update both networks.
The testing function takes the predictions of the new policy of the actor model after the update_epoch to play 10 times and average the 10 scores. If that average is greater than the early value function it stops the code.

Stability in the learning process of the PPO is a real problem. Although all seeds are sets it has been challenging to have replicable results. As a good policy can easily be forgotten by the model, the model saves itself each time he beats his best score, to keep only the best policies.


### Pendulum-v0:

In training 

TRAINED:

![PPO1_GIF_PENDULUM](https://user-images.githubusercontent.com/63811972/152555427-c7b0be3d-8e8d-4638-96a6-090d254c098f.gif)

Plots:

![Pendulum-v0_PPO1_plot](https://user-images.githubusercontent.com/63811972/152562243-f187670b-cdf4-4939-9a2a-7be029ad5e8c.png)



### BipedalWalker-v3

TRAINED:

![GIF_BIPED_289](https://user-images.githubusercontent.com/63811972/152756433-bf651178-1d3b-4776-8dbf-f936b7418bcb.gif)


Plots:
 
![BipedalWalker-v3_plot_PPO_1_DONE](https://user-images.githubusercontent.com/63811972/152756513-1caa4004-46f3-491e-a308-c46220502a4d.png)




## PPO_Batch

The implementation of the second PPO model follows the first one in all point but instead of taking several trajectories with N steps, takes a number of steps (here 256) and divide it into mini-batches that are randomized and fed to the Actor/Critic models.

Hyperparams: 
- Horizon (number of steps the agent takes to update itself at each learning iteration): 256
- mini batches size: 64
- Update_epoch (number of epochs of update of the model): 5
- Learning rate: 0.0003
- Gamma : 0.99
- Clipping range: 0.2
- Entropy Coefficient: 0.01
- Early stop (depends on the environment).


### Pendulum-v0:

TRAINED:


![gif_ppo_mod_intermediate5626trained](https://user-images.githubusercontent.com/63811972/152566900-fd35e396-ab31-4211-9ed4-af0921f0f6ca.gif)



Plots:

![Pendulum-v0_PPO_BATCH_plot](https://user-images.githubusercontent.com/63811972/152566707-7c4e1d0e-5102-48e7-aab6-7c73ccd3ae77.png)



### Links 
* https://spinningup.openai.com/en/latest/algorithms/ppo.html
* https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
* https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
* https://towardsdatascience.com/a-graphic-guide-to-implementing-ppo-for-atari-games-5740ccbe3fbc
* https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
* https://www.youtube.com/watch?v=5P7I-xPq8u8
