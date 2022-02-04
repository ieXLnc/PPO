import matplotlib.pyplot as plt

def gif(images, name, address="./recordings/"):
    images[0].save(address + name, save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)


def plot(rewards_list, actor_loss_list, critic_loss_list, name_task):

    # Create two plots: one for the loss value, one for the accuracy
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

    # Plot accuracy values
    ax1.plot(rewards_list, label='Mean rewards', color='red')
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

    ax3.plot(critic_loss_list, label='Critic Losses', color='yellow')
    ax3.set_title('Critic Network Losses for the {} task'.format(name_task))
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Losses')
    ax3.legend()

    plt.savefig("./Plots/" + name_task + '_plot.png')
    plt.show()