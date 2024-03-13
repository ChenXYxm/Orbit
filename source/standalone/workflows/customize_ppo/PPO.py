import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the CNN architecture for NatureCNN
class NatureCNN(nn.Module):
    def __init__(self):
        super(NatureCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.Dropout(p=0.1, inplace=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Dropout(p=0.1, inplace=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout(p=0.1, inplace=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(123904, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x
class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Dropout(p=0.1, inplace=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout(p=0.1, inplace=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(82944, 512),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x
# Define the CustomPolicy network
class CustomPolicy(nn.Module):
    def __init__(self):
        super(CustomPolicy, self).__init__()
        self.features_extractor = NatureCNN()
        self.mask_feature_extractor = MaskCNN()
        self.pi_features_extractor = NatureCNN()
        self.vf_features_extractor = NatureCNN()
        self.mlp_extractor = MlpExtractor()
        self.action_net = nn.Linear(64, 100)
        self.value_net = nn.Linear(64, 1)

    def forward(self, x,mask):
        x = self.features_extractor(x)
        mask_x = self.mask_feature_extractor(mask)
        combined_features = torch.cat((x, mask_x), dim=1)
        pi = self.mlp_extractor.policy_net(combined_features)
        value = self.mlp_extractor.value_net(combined_features)
        action = self.action_net(value)
        return pi, value, action

# Define the MLP extractor for MlpExtractor
class MlpExtractor(nn.Module):
    def __init__(self):
        super(MlpExtractor, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 32),
            nn.ELU(alpha=1.0),
            nn.Linear(32, 32),
            nn.ELU(alpha=1.0),
            nn.Linear(32, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 64),
            nn.ELU(alpha=1.0)
        )
        self.value_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 32),
            nn.ELU(alpha=1.0),
            nn.Linear(32, 32),
            nn.ELU(alpha=1.0),
            nn.Linear(32, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 64),
            nn.ELU(alpha=1.0)
        )

# Define the PPO agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, epsilon_clip=0.2):
        self.policy_network = CustomPolicy()
        self.optimizer_actor = optim.Adam(self.policy_network.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.policy_network.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            pi, _, _ = self.policy_network(state_tensor, state_tensor)
            action_probs = nn.functional.softmax(pi, dim=1).numpy()
        action = np.random.choice(np.arange(len(action_probs[0])), p=action_probs[0])
        return action

    def update(self, states, actions, rewards, next_states, dones, advantages, returns):
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)

        # Calculate old log probabilities
        with torch.no_grad():
            pi, _, _ = self.policy_network(states_tensor, states_tensor)
            old_log_probs = nn.functional.log_softmax(pi, dim=1).gather(1, actions_tensor.unsqueeze(1))

        for _ in range(10):  # Update policy for 10 epochs
            pi, _, _ = self.policy_network(states_tensor, states_tensor)
            log_probs = nn.functional.log_softmax(pi, dim=1).gather(1, actions_tensor.unsqueeze(1))
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
            surrogate_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor)
            actor_loss = surrogate_loss.mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        # Update value function
        for _ in range(10):  # Update value function for 10 epochs
            _, value, _ = self.policy_network(states_tensor, states_tensor)
            value_loss = nn.functional.mse_loss(value.squeeze(), returns_tensor)
            
            self.optimizer_critic.zero_grad()
            value_loss.backward()
            self.optimizer_critic.step()

# Example usage of the PPOAgent
# Create PPO agent
agent = PPOAgent(state_dim=4, action_dim=2)

# Example training loop
for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        # Calculate advantage and return (you need to implement this)
        advantages, returns = calculate_advantage_and_return(rewards, dones, values)
        agent.update(state, action, reward, next_state, done, advantages, returns)
        state = next_state
