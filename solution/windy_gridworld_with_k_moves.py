import argparse
from sarsa import Sarsa

def get_cmd_args():
	parser = argparse.ArgumentParser(description = 'Windy Gridworld with King\'s Move Simulator.')

	parser.add_argument(
		'-a',
		'--alpha',
		dest = 'learning_rate',
		type = float,
		default = 0.5,
		help = 'It defines the learning rate, and the default value is 0.5')

	parser.add_argument(
		'-g',
		'--gamma',
		dest = 'discount_rate',
		type = float,
		default = 0.9,
		help = 'It defines the discount rate, and the default value is 0.9')

	parser.add_argument(
		'-e',
		'--epsilon',
		dest = 'greedy_rate',
		type = float,
		default = 0.1,
		help = 'It defines the greedy rate, and the default value is 0.1')

	parser.add_argument(
		'-n',
		'--number',
		dest = 'actions_number',
		type = int,
		choices = [4, 8, 9],
		default = 8,
		help = 'It defines the number of actions, and the default value is 8')

	parser.add_argument(
		'-c',
		'--width',
		dest = 'gridworld_width',
		type = int,
		default = 10,
		help = 'It defines the width of the gridworld, and the default value is 10')

	parser.add_argument(
		'-r',
		'--height',
		dest = 'gridworld_height',
		type = int,
		default = 7,
		help = 'It defines the height of the gridworld, and the default value is 7')

	parser.add_argument(
		'-i',
		'--iteration',
		dest = 'episode_number',
		type = int,
		default = 170,
		help = 'It defines the count of episodes, and the default value is 170')

	return parser.parse_args()

if __name__ == '__main__':
	args = get_cmd_args()

	alpha = args.learning_rate
	gamma = args.discount_rate
	epsilon = args.greedy_rate

	actions_number = args.actions_number
	gridworld_height = args.gridworld_height
	gridworld_width = args.gridworld_width

	episode_number = args.episode_number

	background_introduction = '''
	----------- Windy Gridworld with King's Moves -----------

		        1. Learning  Rate: \033[1;31m%.2f\033[0m
		        2. Discount  Rate: \033[1;31m%.2f\033[0m
		        3. Greedy    Rate: \033[1;31m%.2f\033[0m
		        4. Action  Number: \033[1;31m%d\033[0m
		        5. Episode Number: \033[1;31m%d\033[0m

	''' %(alpha, gamma, epsilon, actions_number, episode_number)

	print(background_introduction)

	sarsa = Sarsa(alpha, gamma, epsilon, actions_number, gridworld_height, gridworld_width, episode_number)

	sarsa.sarsa()

	

