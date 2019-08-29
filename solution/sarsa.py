import random
import bisect
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pprint

# Wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# Reward for each step
REWARD = -1

START = [3, 0]
GOAL = [3, 7]

class Sarsa(object):
	def __init__(self, alpha, gamma, epsilon, actions_number, world_height, world_width, episode_number):
		self.__alpha = alpha
		self.__gamma = gamma
		self.__epsilon = epsilon
		self.__actions_number = actions_number
		self.__world_height = world_height
		self.__world_width = world_width
		self.__episode_number = episode_number

		if self.__actions_number == 8:
			# ACTIONS = [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_UP_left, ACTION_UP_RIGHT, ACTION_DOWN_RIGHT, ACTION_DOWN_LEFT]
			self.__action_values = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0),
					 	 5: (-1, -1), 6: (-1, 1), 7: (1, 1), 8: (1, -1)}
		elif self.__actions_number == 9:
			# ACTIONS = [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_UP_left, ACTION_UP_RIGHT, ACTION_DOWN_RIGHT, ACTION_DOWN_LEFT, ACTION_STILL]
			self.__action_values = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0),
					 	 5: (-1, -1), 6: (-1, 1), 7: (1, 1), 8: (1, -1), 9: (0, 0)}
		else:
			# ACTIONS = [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN]
			self.__action_values = {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}

	def build_world(self):
		# Used to represent final policy
		empty_gridworld = [[0 for j in range(self.__world_width)] for i in range(self.__world_height)]
		# Just for testing
		# pprint.pprint(empty_gridworld)
		return empty_gridworld

	def init_q_table(self):
		q_table = {} # key: (s,a), value: value | (s = i,j)
		for a in range(1, self.__actions_number+1):
			for i in range(self.__world_height):
				for j in range(self.__world_width):
					q_table[(i, j, a)] = 0
		# pprint.pprint(q_table)
		return q_table
	

	def find_best_action(self, i, j, q_table):
		candidates = [0] * self.__actions_number

		for k, (ai, aj) in self.__action_values.items():
			out = (i, j, k)
			candidates[k-1] = q_table[out]

		index_opt = np.argmax(candidates)
		value = candidates[index_opt]
		index_opt += 1
		a_i, a_j = self.__action_values[index_opt]
		i_next = max(min(i+a_i, self.__world_height-1), 0)
		j_next = max(min(j+a_j, self.__world_width-1), 0)

		return value, index_opt, (i_next, j_next)

	def choose_action(self, i, j, q_table):
		cdf = [0] * (self.__actions_number+1)

		_, index_opt, candidate_position = self.find_best_action(i, j, q_table)

		for index in range(1, self.__actions_number+1):
			if index == index_opt:
				cdf[index] = cdf[index - 1] + self.__epsilon * 1.0 / self.__actions_number + 1 - self.__epsilon
			else:
				cdf[index] = cdf[index - 1] + self.__epsilon * 1.0 / self.__actions_number

		random_number = random.uniform(0, 1)
		next_action = bisect.bisect_left(cdf, random_number)
		return next_action, candidate_position


	# def plot(self, res, title_name):
	# 	title_name += '_episodes_' + str(self.__episode_number)
	# 	fig = plt.figure()
	# 	plt.plot(res, np.arange(0, self.__episode_number+1))
	# 	plt.title(title_name)
	# 	plt.xlabel('steps')
	# 	plt.ylabel('episodes')
	# 	fig.savefig(title_name + '.png')

	def display_optimal_solution(self, q_table):
		optimal_solution = []
		for i in range(0, self.__world_height):
			optimal_solution.append([])
			for j in range(0, self.__world_width):
				if [i, j] == GOAL:
					optimal_solution[-1].append(' G ')
					continue
				if [i, j] == START:
					optimal_solution[-1].append(' S ')
					continue
				temp = []
				for a in range(1, self.__actions_number):
					temp.append({'action': a, 'value': q_table[(i, j, a)]})
				best_action = max(temp, key = lambda x: x['value'])['action']
				if best_action == 1:
					optimal_solution[-1].append(' L ')
				elif best_action == 2:
					optimal_solution[-1].append(' U ')
				elif best_action == 3:
					optimal_solution[-1].append(' R ')
				elif best_action == 4:
					optimal_solution[-1].append(' D ')
				elif best_action == 5:
					optimal_solution[-1].append('U L')
				elif best_action == 6:
					optimal_solution[-1].append('U R')
				elif best_action == 7:
					optimal_solution[-1].append('D R')
				elif best_action == 8:
					optimal_solution[-1].append('D L')
				elif best_action == 9:
					optimal_solution[-1].append(' N ')
		print('\nOptimal policy is: \n')
		for row in optimal_solution:
			print(row)
		print('Wind strength for each column:\n{}'.format([' ' + str(w) + ' ' for w in WIND]))


	def display_q_table(self, q_table):
		table_header_note = 'Position (Row, Column) '
		table_header_action_basic = '| Left | Up | Right | Down '
		table_header_action_diagonal = '| Up Left | Up Right | Down Right | Down Left '
		table_header_action_still = '| Still'

		if self.__actions_number == 8:
			table_header_action = table_header_note + table_header_action_basic + table_header_action_diagonal
		elif self.__actions_number == 9:
			table_header_action = table_header_note + table_header_action_basic + table_header_action_diagonal + table_header_action_still
		else:
			table_header_action = table_header_note + table_header_action_basic

		pretty_q_table = PrettyTable(table_header_action.split('|'))

		for i in range(self.__world_height):
			for j in range(self.__world_width):
				temp_row = []
				temp_row.append((i, j))
				for a in range(1, self.__actions_number + 1):
					temp_row.append(round(q_table[(i, j, a)], 4))

				pretty_q_table.add_row(temp_row)
				# print(len(temp_row))
				# pprint.pprint(temp_row)
		print('\nFinal Q Table is: \n')
		print(pretty_q_table)

	def sarsa(self):
		state_values = self.build_world()
		optimal_actions = self.build_world()
		q_table = self.init_q_table()

		res = [0]
		i, j = START
		g_i, g_j = GOAL

		print('Starting SARSA...')

		for episode in range(self.__episode_number):
			optimal_actions = self.build_world()
			step_number = 0
			i, j = START
			a, candidate_position = self.choose_action(i, j, q_table)

			while i != g_i or j != g_j:
				# Take the action
				optimal_actions[i][j] = a
				a_i, a_j = candidate_position
				a_i = max(0, a_i - WIND[a_j])

				# Find the next action
				a_next, candidate_position = self.choose_action(a_i, a_j, q_table)
				value = q_table[(i, j, a)]
				
				q_table[(i, j, a)] = value + self.__alpha * (REWARD + self.__gamma * q_table[(a_i, a_j, a_next)] - value)
			
				
				i, j = a_i, a_j
				a = a_next
				
				step_number += 1
			
			res.append(step_number + res[-1])

		self.display_optimal_solution(q_table)
		
		# pprint.pprint(q_table)

		# self.plot(res, 'SARSA_' + str(self.__actions_number) + '_epsilon_' + str(self.__epsilon) + '_gamma_' + str(self.__gamma))

		print('\nOptimal solution is: \n')
		optimal_step = sum(1 for i in optimal_actions for j in i if j > 0)
		print(str(optimal_step) + '\n')
		for i, row in enumerate(optimal_actions):
			print(row)
		print('Wind strength for each column:\n{}'.format([w for w in WIND]))

		self.display_q_table(q_table)
