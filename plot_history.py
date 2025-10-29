import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curve(scores, epsilons, filename, lines=None):
    assert len(scores) % 25 == 0

    means, maxes, mins = [], [] , []
    for i in range(0, len(scores), 25):
        block = np.array(scores[i:i+25])
        means.append(block.mean())
        maxes.append(block.max())
        mins.append(block.min())

    x = range(len(scores)//25)

    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(range(len(epsilons)), epsilons, color="C3")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Epsilon", color="C3")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', colors="C3")

    ax2.scatter(x, means, label = 'mean', color="C1")
    ax2.scatter(x, mins, label = 'min', color="C2")
    ax2.scatter(x, maxes, label = 'max', color="C0")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")
    ax2.legend()

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    
agent = 'DQNAgent'
eps = np.load(os.path.join("models/", f"eps_hist-{agent}.npy"))
scores = np.load(os.path.join("models/", f"scores-{agent}.npy"))
plot_learning_curve(scores, eps, f"plots/{agent}-{len(eps)}")
