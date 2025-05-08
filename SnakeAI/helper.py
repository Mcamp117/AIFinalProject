import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # extra information on plot
    plt.figtext(0.15, 0.01, f'Current Score: {scores[-1]}', fontsize=10)
    plt.figtext(0.65, 0.01, f'Record: {max(scores)}', fontsize=10)

    plt.show(block=False)
    plt.pause(.1)