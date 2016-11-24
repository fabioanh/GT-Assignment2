#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import logging
import copy
import random

logging.basicConfig(level=logging.DEBUG)

# Cooperate Value
coop = 'C'

# Defect value
dfct = 'D'

# Von Newmann type
vonNewmann = 'vonNewmann'

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
        #logging.debug('Creating instance of a player')
        self.neighbourhood = np.empty([neighbourhoodSize])
        self.strategy = strategy
        self.game = game
        self.strategyHistory = [self.strategy]
        #logging.debug('Instance of player created successfully')

    def play(self):
        """
        Runs the game with all of the neighbours and computes the payoff for the current iteration.
        """

        vSinglePlay = np.vectorize(singlePlay)
        payoff = np.sum(vSinglePlay(neighbourhood))

    def singlePlay(self, neighbour):
        return self.game.run([self.strategy, neighbour.strategy])

    def imitate(self):
        # If max function doesn't work, write a function with a loop
        # get index of max in the array and then get the strategy
        self.strategy = max(player.payoff for player in
                            self.neighbourhood).strategy
        self.strategyHistory.append(self.strategy)


class Game:
    """Common base class for all games"""

    # Definition of elements for a game
    numPlayers = 2  # Number of players. Default 2
    matrix = None  # Game Matrix
    strategies = None  # Possible strategy values for the game. Stored as a dictionary with each entry containing [value, index]. The index corresponds to the one in the matrix of the game

    def __init__(self, numPlayers, matrix, strategies):
        logging.debug('Creating instance of game')
        self.numPlayers = numPlayers
        self.matrix = matrix
        self.strategies = strategies
        logging.debug('Instance of game created')

    def run(self, strategies):
        """Executes the current game. Given the value of the game matrix and strategies chosen returns the value for both players"""

        logging.debug('Playing a game')
        return self.matrix[self.strategies[strategies[0]],
                           self.strategies[strategies[1]]]


class Simulator:
    """Simulator class in charge of executing the main logic of the application"""

    # Definition of elements for the simulator
    lattice = None
    game = None
    avgValue = None  # Value used in the terminate computation
    lastLatticeStrategy = None

    def __init__(self, latticeSize, game, neighbourhoodSize, neighbourhoodType, avgValue):
        logging.info('Creating instance of simulator')
        self.initLattice(latticeSize, neighbourhoodSize)
        self.computeNeighbourhoods(neighbourhoodType, latticeSize)
        self.game = game
        self.avgValue = avgValue
        logging.info('Instance of simulator created successfully')

    def initLattice(self, latticeSize, neighbourhoodSize):
        """Initialize the lattice with a set of nxn different players"""
        logging.debug('Initializing lattice for simulator')
        self.lattice = np.empty([latticeSize, latticeSize], dtype=object)
        for i in range(0, latticeSize):
            for j in range(0, latticeSize):
                self.lattice[i, j] = Player(neighbourhoodSize,
                        self.randomStrategy(), self.game)
        logging.debug('Players created in lattice for simulator')

    def randomStrategy(self):
        if random.uniform(0, 1) < 0.5:
            return coop
            return dfct

    def computeNeighbourhoods(self, neighbourhoodType, latticeSize):
        """Initialize the neighbourhoods for the players of the simulation"""
        logging.debug('Computing neighbours for players in lattice')
        for i in range(latticeSize):
            for j in range(latticeSize):
                self.lattice[i, j].neighbourhood = self.computeNeighbours(i, j, len(self.lattice), neighbourhoodType, latticeSize)
        logging.debug('Neighbours successfully assigned for players in lattice')

    def computeNeighbours(self, row, col, size, neighbourhoodType, latticeSize):
        #logging.debug('Computing neighbours for player' + str(row) + ',' + str(col) + ' in lattice')
        neighbours = None
        if neighbourhoodType == vonNewmann:
            neighbours = np.empty([4], dtype=object)
            neighbours[0] = copy.copy(self.lattice[row % latticeSize][(col - 1) % latticeSize])
            neighbours[1] = copy.copy(self.lattice[row % latticeSize][(col + 1) % latticeSize])
            neighbours[2] = copy.copy(self.lattice[(row - 1) % latticeSize][col % latticeSize])
            neighbours[3] = copy.copy(self.lattice[(row + 1) % latticeSize][col % latticeSize])
        if neighbourhoodType == moore:
            neighbours = np.empty([8], dtype=object)
            neighbours[0] = copy.copy(self.lattice[(row - 1) % latticeSize][(col - 1) % latticeSize])
            neighbours[1] = copy.copy(self.lattice[(row - 1) % latticeSize][col % latticeSize])
            neighbours[2] = copy.copy(self.lattice[(row - 1) % latticeSize][(col + 1) % latticeSize])
            neighbours[3] = copy.copy(self.lattice[row % latticeSize][(col - 1) % latticeSize])
            neighbours[4] = copy.copy(self.lattice[row % latticeSize][(col + 1) % latticeSize])
            neighbours[5] = copy.copy(self.lattice[(row + 1) % latticeSize][(col - 1) % latticeSize])
            neighbours[6] = copy.copy(self.lattice[(row + 1) % latticeSize][col % latticeSize])
            neighbours[7] = copy.copy(self.lattice[(row + 1) % latticeSize][(col + 1) % latticeSize])
        return neighbours
    
    def currentLatticeStrategy(self):
        return [p.strategy for p in self.lattice.flat]

    def terminate(self, loop):
        """Determine whether a stable state has been reached and it's good to stop"""
        return self.lastLatticeStrategy == self.currentLatticeStrategy() and loop > 0

    def run(self):
        logging.info('Starting to run simulator')
        
        loop = 0
        
        self.lastLatticeStrategy = self.currentLatticeStrategy()
        
        logging.debug(self.lattice[0][0].payoff)
                      
        map(lambda p: p.play(), self.lattice)
        
        logging.debug(self.lattice[0][0].payoff)
        
        while not self.terminate(loop):
            # check if this works with numpy array otherwise use vectorize or try to call directly lattice.play()
            map(lambda p: p.imitate(), self.lattice)
            map(lambda p: p.play(), self.lattice)
            self.lastLatticeStrategy = self.currentLatticeStrategy()
            logging.info('Iteration: '+ str(loop))
            loop = loop + 1


avgVal = 100  # Average value used to measure the level of cooperation
size = 50  # Latice size
temp = 10  # Temptation payoff
rwrd = 7  # Reward payoff
suck = 0  # Sucker's payoff
pnsh = 0  # Punishment payoff

logging.info('HELLO')
prisionersDilemmaGame = Game(2, np.array([((rwrd, rwrd), (suck, temp)),((temp, suck), (pnsh, pnsh))]), {coop: 0, dfct: 1})
sim = Simulator(size, prisionersDilemmaGame, mooreSize, moore, avgVal)
sim.run()

temp = 10  # Temptation payoff
rwrd = 7  # Reward payoff
suck = 3  # Sucker's payoff
pnsh = 0  # Punishment payoff

snowdriftGame = Game(2, np.array([((rwrd, rwrd), (suck, temp)), ((temp, suck), (pnsh, pnsh))]), {coop: 0, dfct: 1})
sim = Simulator(size, snowdriftGame, mooreSize, moore, avgVal)