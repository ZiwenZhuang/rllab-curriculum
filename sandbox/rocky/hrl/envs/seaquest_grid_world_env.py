from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env, Step
import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.spaces.product import Product

EMPTY = 0
AGENT = 1
DIVER = 2
BOMB = 3
N_OBJECT_TYPES = 4


class GridPlot(object):
    def __init__(self, grid_size, title=None):
        fig, ax = plt.subplots()  #
        #  = plt.figure(figsize=(5, 5))
        # ax = plt.axes(aspect=1)

        plt.tick_params(axis='x', labelbottom='off')
        plt.tick_params(axis='y', labelleft='off')
        if title:
            plt.title(title)

        self.grid_size = grid_size
        self.ax = ax
        self.figure = fig
        self.reset_grid()

    def reset_grid(self):
        self.ax.clear()
        self.ax.set_xticks(range(self.grid_size + 1))
        self.ax.set_yticks(range(self.grid_size + 1))
        self.ax.grid(True, linestyle='-', color=(0, 0, 0), alpha=1, linewidth=1)

    def add_text(self, x, y, text, gravity='center', size=10):
        # transform the grid index to coordinate
        x, y = y, self.grid_size - 1 - x
        if gravity == 'center':
            self.ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', size=size)
        elif gravity == 'left':
            self.ax.text(x + 0.05, y + 0.5, text, ha='left', va='center', size=size)
        elif gravity == 'top':
            self.ax.text(x + 0.5, y + 1 - 0.05, text, ha='center', va='top', size=size)
        elif gravity == 'right':
            self.ax.text(x + 1 - 0.05, y + 0.5, text, ha='right', va='center', size=size)
        elif gravity == 'bottom':
            self.ax.text(x + 0.5, y + 0.05, text, ha='center', va='bottom', size=size)
        else:
            raise NotImplementedError

    def color_grid(self, x, y, color, alpha=1.):
        x, y = y, self.grid_size - 1 - x
        self.ax.add_patch(patches.Rectangle(
            (x, y),
            1,
            1,
            facecolor=color,
            alpha=alpha
        ))


class SeaquestGridWorldEnv(Env):
    """
    A simple Seaquest-like grid world game.

    The observation is a 3D array where the last two dimensions encode the coordinate and the first dimension
    encode whether each type of objects is present in this grid.
    """

    def __init__(self, size=10, n_bombs=None, guided_observation=False):
        """
        Create a new Seaquest-like grid world environment.
        :param size: Size of the grid world
        :param n_bombs: Number of bombs on the grid
        :param guided_observation: whether to include additional information in the observation in the form of
               categorical variables. This could potentially simplify the state predictor used in the MI bonus
               evaluator.
        :return:
        """
        self.grid = None
        self.size = size

        if n_bombs is None:
            n_bombs = size / 2
        self.n_bombs = n_bombs
        self.agent_position = None
        self.diver_position = None
        self.guided_observation = guided_observation
        self.diver_picked_up = False
        self.reset()
        self.fig = None

        visual_obs_space = Box(low=0, high=N_OBJECT_TYPES, shape=(N_OBJECT_TYPES, self.size, self.size))
        if guided_observation:
            guided_obs_space = Product(Discrete(self.size), Discrete(self.size), Discrete(2))
            self._observation_space = Product(visual_obs_space, guided_obs_space)
        else:
            self._observation_space = visual_obs_space
        self._action_space = Discrete(4)

    def reset(self):
        # agent starts at top left corner
        self.agent_position = (0, 0)
        while True:
            self.diver_position = tuple(np.random.randint(low=0, high=self.size, size=2))
            self.bomb_positions = [
                tuple(np.random.randint(low=0, high=self.size, size=2))
                for _ in xrange(self.n_bombs)
                ]
            # ensure that there's a path from the agent to the diver
            if self.feasible():
                break
        self.diver_picked_up = False
        return self.get_current_obs()

    def feasible(self):
        # perform floodfill from the diver position
        if self.agent_position in self.bomb_positions:
            return False
        visited = np.zeros((self.size, self.size))
        visited[self.agent_position] = 1
        cur = self.agent_position
        queue = [cur]
        incs = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        while len(queue) > 0:
            node = queue.pop()
            visited[node] = True
            for inc in incs:
                next = tuple(np.clip(np.array(node) + inc, [0, 0], [self.size - 1, self.size - 1]))
                if next not in self.bomb_positions and not visited[next]:
                    queue.append(next)
        return visited[self.diver_position]

    def get_current_obs(self):
        grid = np.zeros((N_OBJECT_TYPES, self.size, self.size), dtype='uint8')
        grid[(AGENT,) + self.agent_position] = 1
        if not self.diver_picked_up:
            grid[(DIVER,) + self.diver_position] = 1
        for bomb_position in self.bomb_positions:
            grid[(BOMB,) + bomb_position] = 1
        if self.guided_observation:
            return (grid, self.agent_position + (int(self.diver_picked_up),))
        return grid

    def step(self, action):
        coords = np.array(self.agent_position)
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.agent_position = tuple(np.clip(
            coords + increments[action],
            [0, 0],
            [self.size - 1, self.size - 1]
        ))
        if self.agent_position in self.bomb_positions:
            return Step(observation=self.get_current_obs(), reward=0, done=True)
        if self.agent_position == self.diver_position:
            self.diver_picked_up = True
        if self.diver_picked_up and self.agent_position[0] == 0:
            return Step(observation=self.get_current_obs(), reward=1, done=True)
        return Step(observation=self.get_current_obs(), reward=0, done=False)

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

    def render(self):
        if self.fig is None:
            self.fig = GridPlot(self.size)
            plt.ion()
        self.fig.reset_grid()
        self.fig.color_grid(self.agent_position[0], self.agent_position[1], 'g')
        self.fig.add_text(self.agent_position[0], self.agent_position[1], 'Agent')
        if not self.diver_picked_up:
            self.fig.color_grid(self.diver_position[0], self.diver_position[1], 'b')
            self.fig.add_text(self.diver_position[0], self.diver_position[1], 'Diver')
        for bomb_position in self.bomb_positions:
            self.fig.color_grid(bomb_position[0], bomb_position[1], 'r')
            self.fig.add_text(bomb_position[0], bomb_position[1], 'Bomb')
        plt.show()
        plt.pause(0.01)

    def action_from_key(self, key):
        if key in ['left', 'down', 'right', 'up']:
            return self.action_from_direction(key)
        return None

    @property
    def matplotlib_figure(self):
        return self.fig.figure

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def action_from_keys(self):
        pass