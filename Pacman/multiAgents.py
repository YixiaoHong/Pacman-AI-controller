# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print("==================================")
        #print("Successors newPos", newPos)
        #print("Successors Score", successorGameState.getScore())
        #print("Successors newFood", newFood[2])
        #for i in range(len(newGhostStates)):
            #print("Successors GhostState#",i,"is", newGhostStates[i].getPosition())
        #print("Successors newScaredTimes", newScaredTimes)


        "*** YOUR CODE HERE ***"
        ##############################
        #######More Functions########
        #############################
        def printMap(maze):
            string = ''
            for col in maze:
                for char in col:
                    string = string + char
                string = string + '\n'

            print(string)
            return 0

        def printArray(array):
            for i in array:
                print(i)
            return 0

        def hasItem(map, i, j, width, height):
            if (i < 0 or j < 0 or i >= width or j >= height or map[i][j] == '#'):
                return False
            else:
                return True

        def scan(map, i, j, width, height):
            x = step[i][j]
            if (hasItem(map, i + 1, j, width, height) == True):
                if (x + 1 < step[i + 1][j]):
                    step[i + 1][j] = x + 1
                    scan(map, i + 1, j, width, height)
            if (hasItem(map, i, j + 1, width, height) == True):
                if (x + 1 < step[i][j + 1]):
                    step[i][j + 1] = x + 1
                    scan(map, i, j + 1, width, height)
            if (hasItem(map, i - 1, j, width, height) == True):
                if (x + 1 < step[i - 1][j]):
                    step[i - 1][j] = x + 1
                    scan(map, i - 1, j, width, height)
            if (hasItem(map, i, j - 1, width, height) == True):
                if (x + 1 < step[i][j - 1]):
                    step[i][j - 1] = x + 1
                    scan(map, i, j - 1, width, height)

        #Get More Staes
        newScore = successorGameState.getScore()
        currentCapsules = successorGameState.getCapsules()

        #print("Initial Score=", newScore)

        #Get Width and Height
        width = 0
        for i in newFood:
            width = width + 1

        height = len(newFood[0])

        #Virables
        global step
        step = [[999 for i in range(height)] for j in range(width)]

        #get Maze Map
        maze = []

        for x in range(width):
            col = []
            for y in range(height):
                col = col + [' ']
            maze = maze + [col]

        for x in range(width):
            for y in range(height):
                if successorGameState.hasWall(x,y) == True:
                    maze[x][y] = '#'

        for x in range(width):
            for y in range(height):
                if successorGameState.hasFood(x,y) == True:
                    maze[x][y] = '*'

        for i in range(len(currentCapsules)):
            maze[currentCapsules[i][0]][currentCapsules[i][1]] = '$'

        """
        for i in range(len(newGhostStates)):
            posX = int(newGhostStates[i].getPosition()[0])
            posY = int(newGhostStates[i].getPosition()[1])
            if successorGameState.hasFood(posX,posY) == True:
                maze[posX][posY] = 'G'
            elif (posX,posY) in currentCapsules:
                maze[posX][posY] = 'G'
            else:
                maze[posX][posY] = 'g'
        """
        if successorGameState.hasFood(newPos[0], newPos[1]) == True:
            maze[newPos[0]][newPos[1]] = 'P'
        elif (newPos[0], newPos[1]) in currentCapsules:
            maze[newPos[0]][newPos[1]] = 'P'
        else:
            maze[newPos[0]][newPos[1]] = 'p'

        #printMap(maze)

        #Calculate Closest Distance to a Food
        pacmanX = newPos[0]
        pacmanY = newPos[1]

        step[pacmanX][pacmanY] = 0
        scan(maze, pacmanX, pacmanY, x, y)

        closeFoods = []

        for steps in range(999):
            find = False
            for i in range(width):
                for j in range(height):
                    if step[i][j] == steps:
                        if maze[i][j] == '*':
                            find = True
                            closeFoods.append((i, j))
            if find == True:
                break
            else:
                find = False
                closeFoods = []

        #print(closeFoods)

        indexSum = 999

        lowIndexCloseFood = []

        for food in closeFoods:
            tempIndexSum = food[0] + food[1]
            if tempIndexSum <= indexSum:
                indexSum = tempIndexSum
                lowIndexCloseFood.append(food)

        if len(lowIndexCloseFood)!=0:
            closestFood = lowIndexCloseFood[0]

            #print("Choose food:",closestFood)

            toX = closestFood[0]
            toY = closestFood[1]

            #if (step[toX][toY] == 999):
                #print("can not solve")
            #else:
                #print("Get to X,Y need step of:", step[toX][toY])

            foodDis = float(step[toX][toY])
            foodValue = float(1/foodDis)

            #print("foodValue=", foodValue)

            newScore = newScore + foodValue

        #Evaluations using Manhattan
        """
        closestFoodManhattan = 999
        for x in range(width):
            for y in range(height):
                if newFood[x][y] == True:
                    #print("current Grid",x,y)
                    currentGrid = [x,y]
                    tempClosestFoodManhattan = util.manhattanDistance(currentGrid, newPos)
                    #print("Dis=",tempClosestFoodManhattan)
                    if tempClosestFoodManhattan <= closestFoodManhattan:
                        closestFoodManhattan = tempClosestFoodManhattan
                        closestFoodIndex = [x,y]

        newScore = newScore + 10/closestFoodManhattan
        
        """

        #Condiser Capsules (eat it if can)

        if newPos in currentCapsules:
            newScore = newScore + 20

        #Avoid Monsters
        monsterDis = []
        for i in range(len(newGhostStates)):
            monsterX=int(newGhostStates[i].getPosition()[0])
            monsterY=int(newGhostStates[i].getPosition()[1])
            currentMonsterDis = step[monsterX][monsterY]
            currentMonsterDistance = float(currentMonsterDis)
            if (newScaredTimes[i]-5) > currentMonsterDistance:
                newScore = newScore + 2*float(1/currentMonsterDistance)
            else:
                if currentMonsterDistance == 1:
                    newScore = newScore - 50
                elif currentMonsterDistance == 0:
                    newScore = newScore - 500

        #print("FInal Score=",newScore)


        return newScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    #Determe the agent indec of next player, 0 stands for Pacman and the rest stands for monsters
    #The "backToMax" flag will flip to True when next player back to Pacman
    def nextAgent(self,gameState, currentAgent):

        global backToMax

        if (currentAgent + 1) == gameState.getNumAgents(): #rotate back to Pacman
            backToMax = True
            return 0
        else: #Monsters
            return currentAgent + 1

    #PickMaxValue for the Node
    def pickMaxValue(self, gameState, currentDepth,targetDepth, currentAgent):

        maxValue = float("-inf")
        pickAction = "Stop"
        finalReturn = "Stop"

        #print("=============================")
        #print("currentDepth=",currentDepth,"TargetDepth=",targetDepth, "CurrentAgent=", currentAgent)

        if gameState.isWin() or gameState.isLose():
            #print("Its a termminate State")
            finalReturn = self.evaluationFunction(gameState)

        else:
            legalActions = gameState.getLegalActions(currentAgent)

            for actions in legalActions:

                #print("Consider action:", action)

                nextAgent = self.nextAgent(gameState, currentAgent)
                successor = gameState.generateSuccessor(currentAgent, actions)
                currentValue = self.pickMiniMaxNodeValue(successor, currentDepth,targetDepth, nextAgent)

                #print("currentValue=", currentValue)
                #print("maxValue=", maxValue)

                if currentValue > maxValue:
                    maxValue = currentValue
                    pickAction = actions

            if currentDepth == 0:
                finalReturn = pickAction
            else:
                finalReturn = maxValue

        #print("finalReturn is :", finalReturn)

        return finalReturn

    # PickMinValue for the Node
    def pickMinValue(self, gameState,currentDepth,targetDepth, currentAgent):

        minValue = float("inf")

        # print("=============================")
        # print("currentDepth=",currentDepth,"TargetDepth=",targetDepth, "CurrentAgent=", currentAgent)

        if gameState.isWin() or gameState.isLose():
            # print("Its a termminate State")
            finalReturn = self.evaluationFunction(gameState)

        else:
            legalActions = gameState.getLegalActions(currentAgent)

            for actions in legalActions:

                # print("Consider action:", action)

                nextAgent = self.nextAgent(gameState, currentAgent)
                successor = gameState.generateSuccessor(currentAgent, actions)
                currentValue = self.pickMiniMaxNodeValue(successor, currentDepth, targetDepth, nextAgent)

                # print("currentValue=", currentValue)
                # print("minValue=", minValue)

                if currentValue < minValue:
                    minValue = currentValue

            finalReturn = minValue

        # print("finalReturn is :", finalReturn)

        return finalReturn

    #Pick The Node Value(Action) using MiniMax Method
    def pickMiniMaxNodeValue(self,gameState,currentDepth,targetDepth,currentAgent):

        global backToMax

        if backToMax == True: #Determine if rotate back to Pacman, if so, add depth by one
            currentDepth = currentDepth + 1
            #print("A new depth reached, depth=",currentDepth)
            backToMax = False

        if currentDepth == targetDepth: #Determine id reaches the limited Depth, if so, output score directly
            #print("Reached limited Depth")
            pickedValue = self.evaluationFunction(gameState)
        else:
            if currentAgent == 0:
                #print("=========")
                #print("Agent 0")
                pickedValue = self.pickMaxValue(gameState,currentDepth,targetDepth,currentAgent)
                #print("Value=", pickedValue)
                #print("=========")
            else:
                #print("=========")
                #print("Agent Monster", currentAgent)
                pickedValue = self.pickMinValue(gameState,currentDepth,targetDepth,currentAgent)
                #print("Value=", pickedValue)
                #print("=========")

        return pickedValue


    #Return the picked action from MiniMax Method
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        global backToMax
        backToMax = False

        currentAgent = 0
        currentDepth = 0
        targetDepth = self.depth
        actionPick = self.pickMiniMaxNodeValue(gameState, currentDepth, targetDepth, currentAgent)

        return actionPick

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    #Determe the agent indec of next player, 0 stands for Pacman and the rest stands for monsters
    #The "backToMax" flag will flip to True when next player back to Pacman
    def nextAgent(self,gameState, currentAgent):

        global backToMax

        if (currentAgent + 1) == gameState.getNumAgents(): #rotate back to Pacman
            backToMax = True
            return 0
        else: #Monsters
            return currentAgent + 1

    #PickMaxValue for the Node
    def pickMaxValue(self, gameState, currentDepth,targetDepth, currentAgent,alpha,beta):

        maxValue = float("-inf")
        pickAction = "Stop"
        finalReturn = "Stop"
        pruned = False

        if gameState.isWin() or gameState.isLose():
            finalReturn = self.evaluationFunction(gameState)

        else:
            legalActions = gameState.getLegalActions(currentAgent)

            for actions in legalActions:

                nextAgent = self.nextAgent(gameState, currentAgent)
                successor = gameState.generateSuccessor(currentAgent, actions)
                currentValue = self.pickAlphaBetaNodeValue(successor, currentDepth,targetDepth, nextAgent,alpha,beta)

                if currentValue > maxValue:
                    maxValue = currentValue
                    pickAction = actions

                if maxValue >= beta:
                    finalReturn = maxValue
                    pruned = True
                    break

                if currentValue> alpha:
                    alpha = currentValue

            if currentDepth == 0:
                finalReturn = pickAction
            else:
                finalReturn = maxValue

        return finalReturn

    # PickMinValue for the Node
    def pickMinValue(self, gameState,currentDepth,targetDepth, currentAgent,alpha,beta):

        minValue = float("inf")
        pruned = False

        if gameState.isWin() or gameState.isLose():
            finalReturn = self.evaluationFunction(gameState)

        else:
            legalActions = gameState.getLegalActions(currentAgent)

            for actions in legalActions:

                nextAgent = self.nextAgent(gameState, currentAgent)
                successor = gameState.generateSuccessor(currentAgent, actions)
                currentValue = self.pickAlphaBetaNodeValue(successor, currentDepth, targetDepth, nextAgent,alpha,beta)

                if currentValue < minValue:
                    minValue = currentValue

                if minValue <= alpha:
                    finalReturn = minValue
                    pruned = True
                    break

                if currentValue < beta:
                    beta = currentValue

            finalReturn = minValue

        return finalReturn

    #Pick The Node Value(Action) using MiniMax Method
    def pickAlphaBetaNodeValue(self,gameState,currentDepth,targetDepth,currentAgent,alpha,beta):

        global backToMax

        if backToMax == True: #Determine if rotate back to Pacman, if so, add depth by one
            currentDepth = currentDepth + 1
            backToMax = False

        if currentDepth == targetDepth: #Determine if reaches the limited Depth, if so, output score directly
            pickedValue = self.evaluationFunction(gameState)
        else:
            if currentAgent == 0:
                 pickedValue = self.pickMaxValue(gameState,currentDepth,targetDepth,currentAgent,alpha,beta)
            else:
                pickedValue = self.pickMinValue(gameState,currentDepth,targetDepth,currentAgent,alpha,beta)

        return pickedValue

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        global backToMax
        backToMax = False

        currentAgent = 0
        currentDepth = 0
        alpha = float("-inf")
        beta = float("inf")
        targetDepth = self.depth
        actionPick = self.pickAlphaBetaNodeValue(gameState, currentDepth, targetDepth, currentAgent,alpha,beta)

        return actionPick

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Determe the agent indec of next player, 0 stands for Pacman and the rest stands for monsters
    # The "backToMax" flag will flip to True when next player back to Pacman
    def nextAgent(self, gameState, currentAgent):

        global backToMax

        if (currentAgent + 1) == gameState.getNumAgents():  # rotate back to Pacman
            backToMax = True
            return 0
        else:  # Monsters
            return currentAgent + 1

    # PickMaxValue for the Node
    def pickMaxValue(self, gameState, currentDepth, targetDepth, currentAgent):

        maxValue = float("-inf")
        pickAction = "Stop"
        finalReturn = "Stop"

        # print("=============================")
        # print("currentDepth=",currentDepth,"TargetDepth=",targetDepth, "CurrentAgent=", currentAgent)

        if gameState.isWin() or gameState.isLose():
            # print("Its a termminate State")
            finalReturn = self.evaluationFunction(gameState)

        else:
            legalActions = gameState.getLegalActions(currentAgent)

            for actions in legalActions:

                # print("Consider action:", action)

                nextAgent = self.nextAgent(gameState, currentAgent)
                successor = gameState.generateSuccessor(currentAgent, actions)
                currentValue = self.pickExpectimaxNodeValue(successor, currentDepth, targetDepth, nextAgent)

                # print("currentValue=", currentValue)
                # print("maxValue=", maxValue)

                if currentValue > maxValue:
                    maxValue = currentValue
                    pickAction = actions

            if currentDepth == 0:
                finalReturn = pickAction
            else:
                finalReturn = maxValue

        # print("finalReturn is :", finalReturn)

        return finalReturn

    # expectedMinValue for the Node
    def expectedMinValue(self, gameState, currentDepth, targetDepth, currentAgent):

        minValue = float("inf")

        # print("=============================")
        # print("currentDepth=",currentDepth,"TargetDepth=",targetDepth, "CurrentAgent=", currentAgent)

        if gameState.isWin() or gameState.isLose():
            # print("Its a termminate State")
            finalReturn = self.evaluationFunction(gameState)


        else:

            sumValue = 0
            actionNumbers = len(gameState.getLegalActions(currentAgent))
            legalActions = gameState.getLegalActions(currentAgent)

            for actions in legalActions:

                # print("Consider action:", action)

                nextAgent = self.nextAgent(gameState, currentAgent)
                successor = gameState.generateSuccessor(currentAgent, actions)
                currentValue = self.pickExpectimaxNodeValue(successor, currentDepth, targetDepth, nextAgent)

                # print("currentValue=", currentValue)
                # print("minValue=", minValue)

                sumValue = sumValue + currentValue

            expectedValue = float(sumValue/actionNumbers)

            finalReturn = expectedValue

        # print("finalReturn is :", finalReturn)

        return finalReturn

    # Pick The Node Value(Action) using MiniMax Method
    def pickExpectimaxNodeValue(self, gameState, currentDepth, targetDepth, currentAgent):

        global backToMax

        if backToMax == True:  # Determine if rotate back to Pacman, if so, add depth by one
            currentDepth = currentDepth + 1
            # print("A new depth reached, depth=",currentDepth)
            backToMax = False

        if currentDepth == targetDepth:  # Determine id reaches the limited Depth, if so, output score directly
            # print("Reached limited Depth")
            pickedValue = self.evaluationFunction(gameState)
        else:
            if currentAgent == 0:

                pickedValue = self.pickMaxValue(gameState, currentDepth, targetDepth, currentAgent)

            else:

                pickedValue = self.expectedMinValue(gameState, currentDepth, targetDepth, currentAgent)


        return pickedValue


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        global backToMax
        backToMax = False

        currentAgent = 0
        currentDepth = 0
        targetDepth = self.depth
        actionPick = self.pickExpectimaxNodeValue(gameState, currentDepth, targetDepth, currentAgent)

        return actionPick

        util.raiseNotDefined()




def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      1)To avoid the Pacman random moving when there is no food nearby, I implimented a function to
        calculate the minimun steps needed to get to a closest food and add a reciprocal value of that steps to the
        scores. In this way, the Pacman is tend to approach the closest food if there is no food right near it

      2) To avoid the Pacman get caught by the monsters, I implimented a function to calculate how many steps the monster is
         awary from the Pacman, if the distance is 0 (means get caught) there will be 1000 marks deduction. If the distance is 1
         (means very close), there will be 500 marks deduction. In this way, the Pacman will tend to be awary from the monsters
         if thery are too close, however, there will be no marks deduction if the monster is more than 1 steps awary, because
         I don't want the Pacman to be afraid of the monster if they are far awary from it.

      3) To encourage the Pacman to eat capsule and try to kill monsters, a bonus mark (50) is added to the score if the state
         includes Pacman eating a capsule. Also, a reciprocal value of the distance from a scared monster is added to the score
         so that the Pacman will chase the scared monster to kill them and get more score
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    #print("==================================")
    #print("currentPos", currentPos)
    # print("current Score", currentGameState.getScore())
    # print("currentFood", currentFood[2])
    # for i in range(len(currentGhostStates)):
    # print("GhostState#",i,"is", currentGhostStates[i].getPosition())
    #print("Successors currentScaredTimes", currentScaredTimes)

    "*** YOUR CODE HERE ***"

    ##############################
    #######More Functions########
    #############################
    def printMap(maze):
        string = ''
        for col in maze:
            for char in col:
                string = string + char
            string = string + '\n'

        print(string)
        return 0

    def printArray(array):
        for i in array:
            print(i)
        return 0

    def hasItem(map, i, j, width, height):
        if (i < 0 or j < 0 or i >= width or j >= height or map[i][j] == '#'):
            return False
        else:
            return True

    def scan(map, i, j, width, height):
        x = step[i][j]
        if (hasItem(map, i + 1, j, width, height) == True):
            if (x + 1 < step[i + 1][j]):
                step[i + 1][j] = x + 1
                scan(map, i + 1, j, width, height)
        if (hasItem(map, i, j + 1, width, height) == True):
            if (x + 1 < step[i][j + 1]):
                step[i][j + 1] = x + 1
                scan(map, i, j + 1, width, height)
        if (hasItem(map, i - 1, j, width, height) == True):
            if (x + 1 < step[i - 1][j]):
                step[i - 1][j] = x + 1
                scan(map, i - 1, j, width, height)
        if (hasItem(map, i, j - 1, width, height) == True):
            if (x + 1 < step[i][j - 1]):
                step[i][j - 1] = x + 1
                scan(map, i, j - 1, width, height)

    # Get More Staes
    score = currentGameState.getScore()
    currentCapsules = currentGameState.getCapsules()

    #print("Initial Score=", score)

    # Get Width and Height
    width = 0
    for i in currentFood:
        width = width + 1

    height = len(currentFood[0])

    # Virables
    global step
    step = [[999 for i in range(height)] for j in range(width)]

    # get Maze Map
    maze = []

    for x in range(width):
        col = []
        for y in range(height):
            col = col + [' ']
        maze = maze + [col]

    for x in range(width):
        for y in range(height):
            if currentGameState.hasWall(x, y) == True:
                maze[x][y] = '#'

    for x in range(width):
        for y in range(height):
            if currentGameState.hasFood(x, y) == True:
                maze[x][y] = '*'

    for i in range(len(currentCapsules)):
        maze[currentCapsules[i][0]][currentCapsules[i][1]] = '$'

    """
    for i in range(len(currentGhostStates)):
        posX = int(currentGhostStates[i].getPosition()[0])
        posY = int(currentGhostStates[i].getPosition()[1])
        if currentGameState.hasFood(posX,posY) == True:
            maze[posX][posY] = 'G'
        elif (posX,posY) in currentCapsules:
            maze[posX][posY] = 'G'
        else:
            maze[posX][posY] = 'g'
    """
    if currentGameState.hasFood(currentPos[0], currentPos[1]) == True:
        maze[currentPos[0]][currentPos[1]] = 'P'
    elif (currentPos[0], currentPos[1]) in currentCapsules:
        maze[currentPos[0]][currentPos[1]] = 'P'
    else:
        maze[currentPos[0]][currentPos[1]] = 'p'

    #printMap(maze)

    # Calculate Closest Distance to a Food
    pacmanX = currentPos[0]
    pacmanY = currentPos[1]

    step[pacmanX][pacmanY] = 0
    scan(maze, pacmanX, pacmanY, x, y)

    closeFoods = []

    for steps in range(999):
        find = False
        for i in range(width):
            for j in range(height):
                if step[i][j] == steps:
                    if maze[i][j] == '*':
                        find = True
                        closeFoods.append((i, j))
        if find == True:
            break
        else:
            find = False
            closeFoods = []

    #print(closeFoods)

    indexSum = 999

    lowIndexCloseFood = []

    for food in closeFoods:
        tempIndexSum = food[0] + food[1]
        if tempIndexSum <= indexSum:
            indexSum = tempIndexSum
            lowIndexCloseFood.append(food)

    if len(lowIndexCloseFood) != 0:
        closestFood = lowIndexCloseFood[0]

        #print("Choose food:", closestFood)

        toX = closestFood[0]
        toY = closestFood[1]

        #if (step[toX][toY] == 999):
            #print("can not solve")
        #else:
            #print("Get to X,Y need step of:", step[toX][toY])

        foodDis = float(step[toX][toY])
        foodValue = float(1 / foodDis)

        #print("foodValue=", foodValue)

        score = score + foodValue

    # Evaluations using Manhattan
    """
    closestFoodManhattan = 999
    for x in range(width):
        for y in range(height):
            if currentFood[x][y] == True:
                #print("current Grid",x,y)
                currentGrid = [x,y]
                tempClosestFoodManhattan = util.manhattanDistance(currentGrid, currentPos)
                #print("Dis=",tempClosestFoodManhattan)
                if tempClosestFoodManhattan <= closestFoodManhattan:
                    closestFoodManhattan = tempClosestFoodManhattan
                    closestFoodIndex = [x,y]

    score = score + 10/closestFoodManhattan

    """

    # Condiser Capsules (eat it if one step away)

    eatCapsule = False
    for scared in currentScaredTimes:
        if scared >= 38:
            eatCapsule = True
    if eatCapsule == True:
        score = score + 50

    # Avoid Monsters
    monsterDis = []
    for i in range(len(currentGhostStates)):
        monsterX = int(currentGhostStates[i].getPosition()[0])
        monsterY = int(currentGhostStates[i].getPosition()[1])
        currentMonsterDis = step[monsterX][monsterY]
        currentMonsterDistance = float(currentMonsterDis)
        if (currentScaredTimes[i] - 5) > currentMonsterDistance:
            score = score + 3 * float(1 / currentMonsterDistance)
        else:
            if currentMonsterDistance == 1:
                score = score - 500
            elif currentMonsterDistance == 0:
                score = score - 1000

    #print("FInal Score=", score)

    return score


    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction