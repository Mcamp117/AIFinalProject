# Snake AI - Deep Q-Learning Snake Game

This project implements a Snake game with an AI agent trained using Deep Q-Learning (DQN) in Python. The agent learns to play Snake by interacting with the game environment and improving its performance over time.

## Features

- **Deep Q-Learning**: Uses a neural network to approximate Q-values and learn optimal actions.
- **Reward Shaping**: Encourages the agent to move closer to food and penalizes collisions.
- **Training Visualization**: Plots the agent's score and mean score over time.
- **User Play Mode**: Play Snake yourself and compare your score to the AI.

## Project Structure

```
<package manager files>
README.md
model/
    model.pth
SnakeAI/
    agent.py
    game_ai.py
    arial.ttf
    helper.py
    model.py
    game_user.py
```

- [`SnakeAI/agent.py`](SnakeAI/agent.py): Main training loop and DQN agent logic.
- [`SnakeAI/game_ai.py`](SnakeAI/game_ai.py): Snake game environment for AI training.
- [`SnakeAI/model.py`](SnakeAI/model.py): DQN neural network and training logic.
- [`SnakeAI/helper.py`](SnakeAI/helper.py): Plotting utilities for training progress.
- [`SnakeAI/game_user.py`](SnakeAI/game_user.py): Play Snake as a human.
- [`model/model.pth`](model/model.pth): Saved model weights (created after training).

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- IPython
- Pygame

Install dependencies with:

```sh
pip install -r requirements.txt
```

## How to Train the AI

Run the following command to start training the AI agent:

```sh
python SnakeAI/agent.py
```

The training progress will be displayed in the console and as a live plot.

## How to Play as a User

To play the Snake game yourself, run:

```sh
python SnakeAI/game_user.py
```

Use the arrow keys to control the snake.

## Notes

- The trained model is saved to `model/model.pth` automatically when a new high score is achieved.
- You can adjust training parameters (memory size, batch size, learning rate) in [`SnakeAI/agent.py`](SnakeAI/agent.py).

---