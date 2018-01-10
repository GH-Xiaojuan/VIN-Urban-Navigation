import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def visual_data_grid(grid, title=None, show=True, bar=False, cmap='viridis'):
    """
    visualize data grid
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    z_min = np.min(grid)
    z_max = np.max(grid)
    c = ax.pcolor(np.flipud(grid), cmap=cmap, vmin=z_min, vmax=z_max)
    if title:
        ax.set_title(title)
    if bar:
        fig.colorbar(c, ax=ax)
    if show:
        plt.show()

    return fig, ax

def visual_input_rewards(Xstate):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'GOAL']
    for i in range(9):
        image, _ = visual_data_grid(Xstate[i], dirs[i], bar=True)
        image.savefig(dirs[i])
    image, _ = visual_data_grid(np.average(Xstate, axis=0), 'ALL', bar=True)
    image.savefig('ALL')

def draw_path(Xs, Ys, pXs, pYs, value_map):
    goal_x = Xs[-1]
    goal_y = Ys[-1]
    print(goal_x, goal_y)
    fig, ax = visual_data_grid(value_map, 'path_predict', show=False)
    plt.plot(Ys + 0.5, Xs + 0.5, 'ro-', markersize=4, label='real route')
    plt.plot(pYs + 0.5, pXs + 0.5, 'b>-', markersize=4, label='predicted route')
    plt.legend(loc='upper right')
    plt.show()

