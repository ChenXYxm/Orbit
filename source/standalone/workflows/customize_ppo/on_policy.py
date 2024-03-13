import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import os
import pickle
import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

OnPolicyAlgorithmSelf = TypeVar("OnPolicyAlgorithmSelf", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        ########################### add by xy
        #print('start_collect_data')
        training_data = dict()
        self.reach_p = 0
        self.reach_p2 = 0
        self.success_r = 0.0
        round_num = 0.0
        self.success_r2 = 0.0
        self.success_r3 = 0.0
        self.success_rate = 0.0
        self.falling_r = 0.0
        self.falling_r2  = 0.0
        self.num_falling = 0.0
        ###########################
        #print('from ppo')
        #print(self._last_obs.shape)
        #print(self._last_obs.dtype)
        #print(np.max(self._last_obs),np.min(self._last_obs))
        #if np.max(self._last_obs)>1:
            #self._last_obs = self._last_obs/255.0
        #obs_tmp =self._last_obs.copy()
        #obs_tmp = np.squeeze(obs_tmp)
        #for i in range(len(obs_tmp)):
            #plt.imshow(obs_tmp[i])
            #plt.show()
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            #print(n_steps)
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            #if np.max(self._last_obs)>1:
                #self._last_obs = self._last_obs/255.0
            #obs_tmp =self._last_obs.copy()
            #obs_tmp = np.squeeze(obs_tmp)
            #for i in range(len(obs_tmp)):
                #plt.imshow(obs_tmp[i])
                #plt.show()
            #print('obs shape')
            #print(self._last_obs.shape)
            _last_obs_ori = self._last_obs.copy() 
            ###### modified by xy Dec 18
            act_app = np.zeros(len(self._last_obs))
            
            #print(act_app)
            ##############################
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #obs_tensor = obs_tensor.rot90(1,[3,2])
                #fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10))
                #ax1.imshow(np.squeeze(obs_tensor[0,1,:,:].cpu().numpy()))
                #ax2.imshow(np.squeeze(obs_tensor[0,0,:,:].cpu().numpy()))
                #plt.show()
                actions, values, log_probs = self.policy(obs_tensor)
                #print('value')
                #print(values)
                for _ in range(3):
                    obs_tensor = obs_tensor.rot90(1,[3,2])
                    actions_tmp, values_tmp, log_probs_tmp = self.policy(obs_tensor)
                    for i in range(len(log_probs_tmp)):
                        if float(values_tmp[i])>float(values[i]):
                            #print('value')
                            #print(float(values_tmp[i]),float(values[i]))
                            actions[i] = actions_tmp[i].clone()
                            values[i] = values_tmp[i].clone()
                            log_probs[i] = log_probs_tmp[i].clone()
                            act_app[i] = _*2+2
                            self._last_obs[i] = obs_tensor.cpu().numpy()[i].copy()
            #print('new value')
            #print(values)
            #print(act_app)
            #obs_tmp =self._last_obs.copy()
            #obs_tmp = np.squeeze(obs_tmp)
            #for i in range(len(obs_tmp)):
                #plt.imshow(obs_tmp[i])
                #plt.show()
                            
            ############################## add by xy Dec 18
            #self.policy.evaluate_actions(obs_tensor,actions)
            ##############################
            actions = actions.cpu().numpy()
            #print('value and log_probs')
            #print(actions,values,log_probs)
            #print(values.size(),log_probs.size())
            ###################### add by xy Dec 18
            #for _ in range(len(actions)):
                
                #action_ten = th.tensor(actions[_],dtype=th.float32).cuda()
                #advantage_ten = self.policy.value_net(obs_tensor)
                #print('advantage')
                #print(advantage_ten)
            
            #######################################
            # Rescale and perform action
            clipped_actions = actions.copy()
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions.copy(), self.action_space.low, self.action_space.high)
            #################################################### 
            ################################# modified by xy
            clipped_actions_origin = clipped_actions.copy()
            #print('policy clipped action')
            #print(clipped_actions_origin)
            for _ in range(len(self._last_obs)):
                if act_app[_] == 2:
                    clipped_actions[_,0] = 42-clipped_actions[_,1]
                    clipped_actions[_,1] = clipped_actions_origin[_,0]
                elif act_app[_] == 4:
                    clipped_actions[_,0] = 42-clipped_actions[_,0]
                    clipped_actions[_,1] = 42-clipped_actions_origin[_,1]
                elif act_app[_] == 6:
                    clipped_actions[_,0] = clipped_actions[_,1]
                    clipped_actions[_,1] = 42-clipped_actions_origin[_,0]
            ################################# changed in Dec 26
            '''
            for _ in range(len(self._last_obs)):
                if float(values[_]) <=-0.063:
                    act_app[_] = 10
            '''
            #print(act_app)
            ##########################################
            
            clipped_actions_new = np.c_[clipped_actions,act_app.T]
            
            #print(clipped_actions_new)
            new_obs, rewards, dones, infos = env.step(clipped_actions_new)
            ######################### add by xy
            #training_data[self.num_timesteps] = [actions.copy(),clipped_actions.copy(),self._last_obs.copy(),new_obs.copy(),rewards.copy(),dones.copy(),infos,env.env.place_success.cpu().numpy(),
                             #env.env.falling_obj.cpu().numpy(),env.extra_obs.copy()]
            for _ in env.env.check_reaching.tolist():
                if _ >= 0.5:
                   self.reach_p += 1.0 
                   
            for _ in env.env.falling_obj.tolist():
                if _ >= 0.5:
                   self.num_falling += 1.0 
            #print('stop pushing')
            #print(env.env.stop_pushing.tolist())
            
            for _ in env.env.stop_pushing.tolist():
                if _ >= 0.5:
                   self.success_r -= 1.0 
                   self.stop_pushing += 1.0
                   print('stop pushing')
                   print(self.stop_pushing)
                   #print('success r')
                   #print(self.success_r)
            #print(rewards)
            #########################
            #print(rewards)
            #print(infos)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    #print('log in rollout')
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
                    round_num +=1.0
                else:
                    if done:
                       self.success_r += 1.0
                       round_num += 1.0
                       #print('success r')
                       #print(self.success_r)
                       #self.reach_p 
                       #print(self.reach_p)

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            #print('values')
            #print(values)
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            for _ in range(3):
                new_obs_tensor = new_obs_tensor.rot90(1,[3,2])
                values_tmp = self.policy.predict_values(new_obs_tensor)
                for i in range(len(values_tmp)):
                    if float(values_tmp[i])>float(values[i]):
                        values[i] = values_tmp[i]
            #print(values)
            #print(new_obs)
            #print(new_obs.size)
            #new_obs_tmp = 
            #for _ in range(3):
                
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
	################################# add by xy
	
        self.success_rate = self.success_r/float(round_num)
        self.falling_r = self.num_falling/float(round_num)
        self.falling_r2 = self.num_falling/float(n_rollout_steps*env.num_envs)
        self.reach_p2 = self.reach_p/float(n_rollout_steps*env.num_envs)
        if self.success_r < 0:
            self.success_r = 0
        self.success_r2 = self.success_r/float(n_rollout_steps*env.num_envs)
        if self.reach_p >0:
            self.success_r3 = self.success_r/self.reach_p
        else:
            self.success_r3
        #################################
        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: OnPolicyAlgorithmSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
    ) -> OnPolicyAlgorithmSelf:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        self.num_timesteps = 336000
        self._num_timesteps_at_start = 336000
        self.falling_r2 = 0.0
        self.falling_r = 0.
        self.reach_p = 0.
        self.success_r = 0.
        self.success_r2 = 0.
        self.success_r3 = 0.
        self.success_rate = 0.
        self.reach_p2 = 0.
        self.stop_pushing = 0.
        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/reach_rate",self.reach_p2)
                self.logger.record("rollout/falling_rate",self.falling_r)
                self.logger.record("rollout/falling_rate_per_step",self.falling_r2)
                self.logger.record("rollout/success_rate",self.success_rate)
                
                self.logger.record("rollout/success_rate2",self.success_r2)
                self.logger.record("rollout/success_vs_pushing",self.success_r3)
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []