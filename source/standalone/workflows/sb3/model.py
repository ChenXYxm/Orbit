import pickle as pkl
from stable_baselines3 import PPO
import numpy as np
import torch
import gym
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy

class CustomEnvironment(gym.Env):
    def __init__(self):
        super(CustomEnvironment, self).__init__()

        # Define observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(50, 50, 2),dtype=np.uint8)

        # Define action space
        self.action_space = gym.spaces.MultiDiscrete([50,50])

        # Initialize state
        self.state = np.zeros(3)

    def step(self, action):
        # Example dynamics: move state in the direction of the selected action
        if action == 0:
            self.state[0] += 0.1
        else:
            self.state[0] -= 0.1

        # Example reward: based on how close the first component of the state is to a target value
        reward = -abs(self.state[0] - 0.5)

        # Example termination: episode terminates after a certain number of steps
        done = False
        if self.state[0] < -1.0 or self.state[0] > 1.0:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state to the initial state
        self.state = np.zeros(3)
        return self.state

# Create custom environment
class CustomPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, features_extractor_class= NatureCNN,activation_fn=nn.ELU,net_arch={'pi': [32,32,256, 128, 64], 'vf': [32,32,256, 128, 64]})
        print(self.net_arch)
        
class CustomMLPExtractor(nn.Module):
    def __init__(self, observation_space, features_dim, activation_fn=nn.ELU):
        super(CustomMLPExtractor, self).__init__()
        self.net_arch = [
            dict(
                pi=[256, 128, 64],
                vf=[256, 128, 64]
            )
        ]
        self.mlp_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            activation_fn(),
            nn.Linear(features_dim, features_dim),
            activation_fn()
        )

    def forward(self, observations):
        return self.mlp_extractor(observations)

class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Flatten(),
            
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))



def main():
    print("GPU Available:", torch.cuda.is_available())
    
    checkpoint = './data/model/weight758080.pth'
    state_dict = torch.load(checkpoint)
    layer_names = state_dict.keys()
    print("Layer Names:")
    for name in layer_names:
        print(name)
    env = DummyVecEnv([lambda: CustomEnvironment()])
    dummy_model = PPO(CustomPolicy, env, verbose=1)
    #dummy_model = PPO('CnnPolicy', env, verbose=1)
    new_policy_state_dict = dummy_model.policy.state_dict()
    print(dummy_model.policy)
    print(dummy_model.policy.device)
    for name, param in dummy_model.policy.named_parameters():
        print(name)
        #print(param)
    new_policy_state_dict["features_extractor.cnn.0.weight"].copy_(state_dict["features_extractor.cnn.0.weight"])
    new_policy_state_dict["features_extractor.cnn.0.bias"].copy_(state_dict["features_extractor.cnn.0.bias"])
    new_policy_state_dict["features_extractor.cnn.2.weight"].copy_(state_dict["features_extractor.cnn.2.weight"])
    new_policy_state_dict["features_extractor.cnn.2.bias"].copy_(state_dict["features_extractor.cnn.2.bias"])
    new_policy_state_dict["features_extractor.cnn.5.weight"].copy_(state_dict["features_extractor.cnn.5.weight"])
    new_policy_state_dict["features_extractor.cnn.5.bias"].copy_(state_dict["features_extractor.cnn.5.bias"])
    new_policy_state_dict["features_extractor.cnn.8.weight"].copy_(state_dict["features_extractor.cnn.8.weight"])
    new_policy_state_dict["features_extractor.cnn.8.bias"].copy_(state_dict["features_extractor.cnn.8.bias"])
    new_policy_state_dict["features_extractor.linear.0.weight"].copy_(state_dict["features_extractor.linear.0.weight"])
    new_policy_state_dict["features_extractor.linear.0.bias"].copy_(state_dict["features_extractor.linear.0.bias"])
    ########### shared layers
    new_policy_state_dict["mlp_extractor.policy_net.0.weight"].copy_(state_dict["mlp_extractor.shared_net.0.weight"])
    new_policy_state_dict["mlp_extractor.policy_net.0.bias"].copy_(state_dict["mlp_extractor.shared_net.0.bias"])
    new_policy_state_dict["mlp_extractor.policy_net.2.weight"].copy_(state_dict["mlp_extractor.shared_net.2.weight"])
    new_policy_state_dict["mlp_extractor.policy_net.2.bias"].copy_(state_dict["mlp_extractor.shared_net.2.bias"])
    
    new_policy_state_dict["mlp_extractor.value_net.0.weight"].copy_(state_dict["mlp_extractor.shared_net.0.weight"])
    new_policy_state_dict["mlp_extractor.value_net.0.bias"].copy_(state_dict["mlp_extractor.shared_net.0.bias"])
    new_policy_state_dict["mlp_extractor.value_net.2.weight"].copy_(state_dict["mlp_extractor.shared_net.2.weight"])
    new_policy_state_dict["mlp_extractor.value_net.2.bias"].copy_(state_dict["mlp_extractor.shared_net.2.bias"])
    ############ policy net
    new_policy_state_dict["mlp_extractor.policy_net.4.weight"].copy_(state_dict["mlp_extractor.policy_net.0.weight"])
    new_policy_state_dict["mlp_extractor.policy_net.4.bias"].copy_(state_dict["mlp_extractor.policy_net.0.bias"])
    new_policy_state_dict["mlp_extractor.policy_net.6.weight"].copy_(state_dict["mlp_extractor.policy_net.2.weight"])
    new_policy_state_dict["mlp_extractor.policy_net.6.bias"].copy_(state_dict["mlp_extractor.policy_net.2.bias"])
    new_policy_state_dict["mlp_extractor.policy_net.8.weight"].copy_(state_dict["mlp_extractor.policy_net.4.weight"])
    new_policy_state_dict["mlp_extractor.policy_net.8.bias"].copy_(state_dict["mlp_extractor.policy_net.4.bias"])
    ############ value net
    new_policy_state_dict["mlp_extractor.value_net.4.weight"].copy_(state_dict["mlp_extractor.value_net.0.weight"])
    new_policy_state_dict["mlp_extractor.value_net.4.bias"].copy_(state_dict["mlp_extractor.value_net.0.bias"])
    new_policy_state_dict["mlp_extractor.value_net.6.weight"].copy_(state_dict["mlp_extractor.value_net.2.weight"])
    new_policy_state_dict["mlp_extractor.value_net.6.bias"].copy_(state_dict["mlp_extractor.value_net.2.bias"])
    new_policy_state_dict["mlp_extractor.value_net.8.weight"].copy_(state_dict["mlp_extractor.value_net.4.weight"])
    new_policy_state_dict["mlp_extractor.value_net.8.bias"].copy_(state_dict["mlp_extractor.value_net.4.bias"])
    new_policy_state_dict["mlp_extractor.value_net.8.weight"].copy_(state_dict["mlp_extractor.value_net.4.weight"])
    new_policy_state_dict["mlp_extractor.value_net.8.bias"].copy_(state_dict["mlp_extractor.value_net.4.bias"])
    new_policy_state_dict["action_net.weight"].copy_(state_dict["action_net.weight"])
    new_policy_state_dict["action_net.bias"].copy_(state_dict["action_net.bias"])
    new_policy_state_dict["value_net.weight"].copy_(state_dict["value_net.weight"])
    new_policy_state_dict["value_net.bias"].copy_(state_dict["value_net.bias"])
    dummy_model.policy.load_state_dict(new_policy_state_dict)
    dummy_model.save("./data/model/model2")
    for name, param in dummy_model.policy.named_parameters():
        print(name)
        print(param)
    
    #checkpoint = './data/model.zip'
    #dummy_model = PPO.load(checkpoint)
    print(f"Loading checkpoint from: {checkpoint}")
    fileObject2 = open('./data/tmp_data2.pkl', 'rb')
    data_real=  pkl.load(fileObject2)
    fileObject2.close()
    print(data_real.shape)
    print(data_real[0,:,:,0])
    print(data_real[0,:,:,1])
    obs = np.zeros(data_real.shape)
    obs = data_real.copy()
    obs = np.rot90(obs,1,(2,1))
    obs = obs.copy()
    obs = np.rot90(obs,1,(2,1))
    obs = obs.copy()
    actions, _ = dummy_model.predict(obs, deterministic=True)
    print(actions)

if __name__ == "__main__":
    main()
