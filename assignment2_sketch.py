import numpy as np
import logging
import copy

# Latice size
n = 50

vonNewmannSize = 4
mooreSize = 8

class Player:
	"""Class representing a player that will take part in the problems"""
	# Definition of elements for a player
	neighbourhood = None
	game = None
	payoff = 0
	strategy = None

	def __init__(self, neighbourhoodSize):
		self.neighbourhood = np.empty([neighbourhoodSize])

	def play(self):
		"""Runs the game with all of the neighbours and computes the payoff for the current iteration"""
		pyoff = 0
		
		pyoff = self.game.run()

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

	def __init__(self, numPlayers, strategies):
		self.numPlayers = numPlayers
		self.strategies = strategies
		self.matrix = np.empty([len(strategies), len(strategies)])
	def __init__(self, strategies):
		self.strategies = strategies
		self.matrix = np.empty([len(strategies), len(strategies)])

	def run(self, strategies):
		"""Executes the current game. Given the value of the game matrix and strategies chosen returns the value for both players"""
		logging.debug('Playing a game')
		return self.matrix[self.strategies[strategies[0]], self.strategies[strategies[1]]]

class Simulator:
	"""Simulator class in charge of executing the main logic of the application"""
	# Definition of elements for the simulator
	lattice = None
	game = None

	def __initi__(self, latticeSize, game, neighbourhoodSize, neighbourhoodType):
		self.lattice = initLattice(latticeSize, neighbourhoodSize)
		self.computeNeighbourhoods(neighbourhoodType, latticeSize)
		self.game = game

	def initLattice(self, latticeSize, neighbourhoodSize):
		"""Initialize the lattice with a set of nxn different players"""
		self.lattice[i,j] = np.empty([latticeSize, latticeSize])
		for i in range(latticeSize):
    		for j in range(latticeSize):
        		self.lattice[i,j] = Player(neighbourhoodSize)
        		self.lattice[i,j].game = self.game

    def computeNeighbourhoods(self, neighbourhoodType, latticeSize):
    	"""Initialize the neighbourhoods for the players of the simulation"""
    	for i in range(latticeSize):
    		for j in range(latticeSize):
    			self.lattice[i,j].neighbourhood = self.computeNeighbours(i, j, len(self.lattice), neighbourhoodType), latticeSize

    def computeNeighbours(self, row, col, size, neighbourhoodType, latticeSize):
    	neighbours = None
    	if neighbourhoodType == 'vonNewmann':
    		neighbours = np.empty([4])
    		neighbours[0] = copy.deepCopy(self.lattice[row % latticeSize][(col - 1) % latticeSize])
    		neighbours[1] = copy.deepCopy(self.lattice[row % latticeSize][(col + 1) % latticeSize])
    		neighbours[2] = copy.deepCopy(self.lattice[(row - 1) % latticeSize][col % latticeSize])
    		neighbours[3] = copy.deepCopy(self.lattice[(row + 1) % latticeSize][col % latticeSize])
    	if neighbourhoodType == 'moore':
    		neighbours = np.empty([8])
    		neighbours[0] = copy.deepCopy(self.lattice[(row - 1) % latticeSize][(col - 1) % latticeSize])
    		neighbours[1] = copy.deepCopy(self.lattice[(row - 1) % latticeSize][col % latticeSize])
    		neighbours[2] = copy.deepCopy(self.lattice[(row - 1) % latticeSize][(col + 1) % latticeSize])
    		neighbours[3] = copy.deepCopy(self.lattice[row % latticeSize][(col - 1) % latticeSize])
    		neighbours[4] = copy.deepCopy(self.lattice[row % latticeSize][(col + 1) % latticeSize])
    		neighbours[5] = copy.deepCopy(self.lattice[(row + 1) % latticeSize][(col - 1) % latticeSize])
    		neighbours[6] = copy.deepCopy(self.lattice[(row + 1) % latticeSize][col % latticeSize])
    		neighbours[7] = copy.deepCopy(self.lattice[(row + 1) % latticeSize][(col + 1) % latticeSize])
    	return neighbours


temp = 10 # Temptation payoff
rwrd = 7 # Reward payoff
suck = 0 # Sucker's payoff
pnsh = 0 # Punishment payoff

prisionersDilemmaGame = Game(2, np.array([((rwrd, rwrd), (suck, temp)), ((temp, suck), (pnsh, pnsh))]), {"C":0, "D":1})

temp = 10 # Temptation payoff
rwrd = 7 # Reward payoff
suck = 3 # Sucker's payoff
pnsh = 0 # Punishment payoff

snowdriftGame = Game(2, np.array([((rwrd, rwrd), (suck, temp)), ((temp, suck), (pnsh, pnsh))]), {"C":0, "D":1})