import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gif(images, name, address="./recordings/"):
    images[0].save(address + name, save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)


def smoothen(x, winsize=20):
    return np.array(pd.Series(x).rolling(winsize).mean())[winsize-1:]


def plot(rewards_list, actor_loss_list, critic_loss_list, losses_t, name_task):

    # Create two plots: one for the loss value, one for the accuracy
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

    # Plot accuracy values
    mean_val = smoothen(rewards_list)
    ax1.plot(rewards_list, label='Mean rewards', color='red', alpha=0.3)
    ax1.plot(mean_val, label='Average smoothed', color='red')
    ax1.set_title('Mean Rewards for the {} task'.format(name_task))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Rewards')
    ax1.legend()

    # Plot accuracy values
    ax2.plot(actor_loss_list, label='Actor Losses', color='green')
    ax2.set_title('Actor Network Losses for the {} task'.format(name_task))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Losses')
    ax2.legend()

    mean_val = smoothen(critic_loss_list)
    ax3.plot(critic_loss_list, label='Critic Losses', color='yellow', alpha = 0.3)
    ax3.plot(mean_val, label='Average smoothed', color='yellow')
    ax3.set_title('Critic Network Losses for the {} task'.format(name_task))
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Losses')
    ax3.legend()

    mean_val = smoothen(losses_t)
    ax4.plot(losses_t, label='General Losses', color='black', alpha=0.3)
    ax4.plot(mean_val, label='Average smoothed', color='black')
    ax4.set_title('General Losses for the {} task'.format(name_task))
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Losses')
    ax4.legend()

    plt.savefig("./Plots/" + name_task + '_plot.png')
    plt.show()