import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, Polygon


class CliffWalkingRenderer:
    actions_to_text = {
        0: '^',
        1: '>',
        2: 'v',
        3: '<',
        None: 'o'  # finish state
    }

    actions_to_text_coordinates = {
        0: (0.3, 0.8),
        1: (0.6, 0.45),
        2: (0.3, 0.1),
        3: (0, 0.45)
    }

    actions_to_triangle_coordinates = {
        0: [[0, 1], [0.5, 0.5], [1, 1]],
        1: [[1, 1], [0.5, 0.5], [1, 0]],
        2: [[0, 0], [0.5, 0.5], [1, 0]],
        3: [[0, 1], [0.5, 0.5], [0, 0]]
    }

    def __init__(self, env, q_values, start_from_episode=1):
        self.env = env
        self.rows = 4
        self.columns = 12
        self.red_cells = [(x, 0) for x in range(1, 11)]
        self.states_to_cells = self._create_states_dict()
        self.cells_to_states = dict(zip(self.states_to_cells.values(), self.states_to_cells.keys()))
        self.q_values = q_values
        self.start_from_episode = start_from_episode
        fig = plt.figure(figsize=(self.columns, self.rows))
        self.ax = fig.add_subplot(111, aspect='equal')
        matplotlib.rcParams.update({'font.size': 7})

    def _create_states_dict(self):
        states = {}
        i = 0
        for y in range(3, -1, -1):
            for x in range(self.columns):
                states[i] = (x, y)
                i += 1
        return states

    def _display_episode_info(self, episode, step_counter, finished):
        if not finished:
            plt.title('Episode {0}, Step: {1}'.format(episode, step_counter))
        else:
            plt.title('Episode {0} finished on step {1}'.format(episode, step_counter))

    def _get_q_value_color_info(self):
        q_values = []
        for x in range(self.columns):
            for y in range(self.rows):
                for cell_action in range(self.env.action_space.n):
                    if ((x, y) not in self.red_cells) & ((x, y) != (11, 0)):
                        cell_state = self.cells_to_states[(x, y)]
                        q_values.append(self.q_values[(cell_state, cell_action)])
        min_q_value = min(q_values)
        max_q_value = max(q_values) + 0.0001  # so min_q_value != max_q_value
        color_norm = matplotlib.colors.Normalize(vmin=min_q_value, vmax=max_q_value)
        colors = [[color_norm(min_q_value), "red"],
                  [color_norm(max_q_value), "lightgreen"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        return cmap, color_norm

    def _draw_q_value_triangle(self, x, y, cell_action, cmap, color_norm, q_value):
        triangle_cell = self.actions_to_triangle_coordinates[cell_action]
        triangle_coordinates = []
        for i in range(len(triangle_cell)):
            triangle_coordinates.append([triangle_cell[i][0] + x, triangle_cell[i][1] + y])

        triangle = Polygon(triangle_coordinates, edgecolor='k', facecolor=cmap(color_norm(q_value)))
        self.ax.add_patch(triangle)

    def _draw_q_value_text(self, x, y, cell_action, q_value):
        text_x, text_y = self.actions_to_text_coordinates[cell_action]
        plt.text(x + text_x, y + text_y, q_value, zorder=1)

    def _draw_reward_cells(self, x, y):
        if (x, y) in self.red_cells:
            rect = Rectangle((x, y), 1, 1, edgecolor='k', facecolor='r', zorder=2)
            self.ax.add_patch(rect)
        elif (x, y) == (11, 0):
            rect = Rectangle((x, y), 1, 1, edgecolor='k', facecolor='lawngreen', zorder=2)
            self.ax.add_patch(rect)

    def _draw_agent(self, state, action):
        player_x, player_y = self.states_to_cells[state]
        plt.scatter(player_x + 0.5, player_y + 0.5, s=60, zorder=3, marker=self.actions_to_text[action],
                    edgecolors='k', facecolor='w')

    def draw_environment(self, state, episode, step_counter, action=None, finished=False, clear_display=True):
        if episode >= self.start_from_episode:
            self._display_episode_info(episode, step_counter, finished)
            cmap, color_norm = self._get_q_value_color_info()

            for x in range(self.columns):
                for y in range(self.rows):
                    self._draw_reward_cells(x, y)
                    for cell_action in range(self.env.action_space.n):
                        cell_state = self.cells_to_states[(x, y)]
                        q_value = round(self.q_values[(cell_state, cell_action)], 2)
                        self._draw_q_value_triangle(x, y, cell_action, cmap, color_norm, q_value)
                        self._draw_q_value_text(x, y, cell_action, q_value)

            self._draw_agent(state, action)

            plt.ylim((0, self.rows))
            plt.xlim((0, self.columns))
            plt.pause(0.0001)
            if clear_display:
                plt.cla()
            else:
                plt.show()
