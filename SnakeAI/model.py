import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, filename='model.pth'):
        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
        torch.save(self.state_dict(), filepath)

class DQNTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.lr = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predict Q-values for current state
        q_pred = self.model(state)
        q_target = q_pred.clone().detach()

        for idx, terminal in enumerate(done):
            q_update = reward[idx]
            if not terminal:
                q_update += self.gamma * torch.max(self.model(next_state[idx])).item()
            q_target[idx][torch.argmax(action[idx]).item()] = q_update

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_pred, q_target)
        loss.backward()
        self.optimizer.step()