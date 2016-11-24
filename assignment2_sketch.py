import numpy as np
import logging
import copy
import random

# Cooperate Value
coop="C"
# Defect value
dfct="D"
# Von Newmann type
vonNewmann='vonNewmann'
# Moore type
moore = 'moore'

vonNewmannSize = 4
mooreSize = 8

class Player:
	"""Class representing a player that will take part in the problems"""
	# Definition of elements for a player
	neighbourhood = None
	game = None
	payoff = 0
	strategy = None
	strategyHistory = None

	def __init__(self, neighbourhoodSize, strategy, game):
		self.neighbourhood = np.empty([neighbourhoodSize])
		self.strategy = strategy
		self.game = game
		self.strategyHistory = [self.strategy]

	def play(self):
		"""
		Runs the game with all of the neighbours and computes the payoff for the current iteration.
		"""
		vSinglePlay = np.vectorize(singlePlay)
		payoff = np.sum(vSinglePlay(neighbourhood))

	def singlePlay(self, neighbour):
		self.game.run([self.strategy, neighbour.strategy])

	def imitate(self):
		#If max function doesn't work, write a function with a loop
		self.strategy = max(player.payoff for player in self.neighbourhood).strategy
		self.strategyHistory.append(self.strategy)

class Game:
	"""Common base class for all games"""
	# Definition of elements for a game
	numPlayers = 2 # Number of players. Default 2
	matrix = None # Game Matrix
	strategies = None # Possible strategy values for the game. Stored as a dictionary with each entry containing [value, index]. The index corresponds to the one in the matrix of the game

	def __init__(self, numPlayers, matrix, strategies):
		self.numPlayers = numPlayers
		self.matrix = matrix
		self.strategies = strategies

	def run(self, strategies):
		"""Executes the current game. Given the value of the game matrix and strategies chosen returns the value for both players"""
		logging.debug('Playing a game')
		return self.matrix[self.strategies[strategies[0]], self.strategies[strategies[1]]]

class Simulator:
	"""Simulator class in charge of executing the main logic of the application"""
	# Definition of elements for the simulator
	lattice = None
	game = None
	avgValue = None # Value used in the terminate computation
	lastStrategy = None

	def __init__(self, latticeSize, game, neighbourhoodSize, neighbourhoodType, avgValue):
		self.initLattice(latticeSize, neighbourhoodSize)
		self.computeNeighbourhoods(neighbourhoodType, latticeSize)
		self.game = game
		self.avgValue = avgValue

	def initLattice(self, latticeSize, neighbourhoodSize):
		"""Initialize the lattice with a set of nxn different players"""
		self.lattice = np.empty([latticeSize, latticeSize], dtype = object)
		for i in range(0,latticeSize):
			for j in range(0,latticeSize):
				self.lattice[i,j] = Player(neighbourhoodSize, self.randomStrategy(), self.game)

	def randomStrategy(self):
		if random.uniform(0,1) < 0.5:
			return coop
			return dfct

	def computeNeighbourhoods(self, neighbourhoodType, latticeSize):
		"""Initialize the neighbourhoods for the players of the simulation"""
		for i in range(latticeSize):
			for j in range(latticeSize):
				self.lattice[i,j].neighbourhood = self.computeNeighbours(i, j, len(self.lattice), neighbourhoodType, latticeSize)

	def computeNeighbours(self, row, col, size, neighbourhoodType, latticeSize):
		neighbours = None
		if neighbourhoodType == vonNewmann:
			neighbours = np.empty([4], dtype = object)
			neighbours[0] = copy.deepcopy(self.lattice[row % latticeSize][(col - 1) % latticeSize])
			neighbours[1] = copy.deepcopy(self.lattice[row % latticeSize][(col + 1) % latticeSize])
			neighbours[2] = copy.deepcopy(self.lattice[(row - 1) % latticeSize][col % latticeSize])
			neighbours[3] = copy.deepcopy(self.lattice[(row + 1) % latticeSize][col % latticeSize])
		if neighbourhoodType == moore:
			neighbours = np.empty([8], dtype = object)
			neighbours[0] = copy.deepcopy(self.lattice[(row - 1) % latticeSize][(col - 1) % latticeSize])
			neighbours[1] = copy.deepcopy(self.lattice[(row - 1) % latticeSize][col % latticeSize])
			neighbours[2] = copy.deepcopy(self.lattice[(row - 1) % latticeSize][(col + 1) % latticeSize])
			neighbours[3] = copy.deepcopy(self.lattice[row % latticeSize][(col - 1) % latticeSize])
			neighbours[4] = copy.deepcopy(self.lattice[row % latticeSize][(col + 1) % latticeSize])
			neighbours[5] = copy.deepcopy(self.lattice[(row + 1) % latticeSize][(col - 1) % latticeSize])
			neighbours[6] = copy.deepcopy(self.lattice[(row + 1) % latticeSize][col % latticeSize])
			neighbours[7] = copy.deepcopy(self.lattice[(row + 1) % latticeSize][(col + 1) % latticeSize])
			return neighbours

	def terminate():
		"""Determine whether a stable state has been reached and it's good to stop"""
		return 

	def run(self):
		while(not self.terminate()):
		#check if this works with numpy array otherwise use vectorize or try to call directly lattice.play()
			map(lambda p : p.imitate(), lattice)
			map(lambda p : p.play(), lattice)



avgVal = 100 # Average value used to measure the level of cooperation
size = 50 # Latice size
temp = 10 # Temptation payoff
rwrd = 7 # Reward payoff
suck = 0 # Sucker's payoff
pnsh = 0 # Punishment payoff

prisionersDilemmaGame = Game(2, np.array([((rwrd, rwrd), (suck, temp)), ((temp, suck), (pnsh, pnsh))]), {coop:0, dfct:1})
sim = Simulator(size, prisionersDilemmaGame, mooreSize, moore, avgVal)

temp = 10 # Temptation payoff
rwrd = 7 # Reward payoff
suck = 3 # Sucker's payoff
pnsh = 0 # Punishment payoff

snowdriftGame = Game(2, np.array([((rwrd, rwrd), (suck, temp)), ((temp, suck), (pnsh, pnsh))]), {coop:0, dfct:1})
sim = Simulator(size, snowdriftGame, mooreSize, moore)
