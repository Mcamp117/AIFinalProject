import torch
import random
import numpy as np
from collections import deque
from aiGame import SnakeGameAI, Direction, Point
from LilHelper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def getState(self, game):
        head = game.snake[0]
        pointLeft = Point(head.x - 20, head.y)
        pointRight = Point(head.x + 20, head.y)
        pointUp = Point(head.x, head.y - 20)
        pointDown = Point(head.x, head.y + 20)
        
        dirLeft = game.direction == Direction.LEFT
        dirRight = game.direction == Direction.RIGHT
        dirUp = game.direction == Direction.UP
        dirDown = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dirRight and game.is_collision(pointRight)) or 
            (dirLeft and game.is_collision(pointLeft)) or 
            (dirUp and game.is_collision(pointUp)) or 
            (dirDown and game.is_collision(pointDown)),

            # Danger right
            (dirUp and game.is_collision(pointRight)) or 
            (dirDown and game.is_collision(pointLeft)) or 
            (dirLeft and game.is_collision(pointUp)) or 
            (dirRight and game.is_collision(pointDown)),

            # Danger left
            (dirDown and game.is_collision(pointRight)) or 
            (dirUp and game.is_collision(pointLeft)) or 
            (dirRight and game.is_collision(pointUp)) or 
            (dirLeft and game.is_collision(pointDown)),
            
            # Move direction
            dirLeft,
            dirRight,
            dirUp,
            dirDown,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def trainLongMem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def trainShortMem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def getAction(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        finalMove = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove


def train():
    plotScores = []
    plotMeanScores = []
    totScore = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        prevState = agent.get_state(game)

        # get move
        finalMove = agent.get_action(prevState)

        # perform move and get new state
        reward, done, score = game.play_step(finalMove)
        state_new = agent.get_state(game)

        # train short memory
        agent.trainShortMem(prevState, finalMove, reward, state_new, done)

        # remember
        agent.remember(prevState, finalMove, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.trainLongMem()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plotScores.append(score)
            totScore += score
            meanScore = totScore / agent.n_games
            plotMeanScores.append(mean_score)
            plot(plotScores, plotMeanScores)


if __name__ == '__main__':
    train()