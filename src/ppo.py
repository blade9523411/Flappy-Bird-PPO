import gym
import time 
import numpy as np
import time
import torch
import pygame 
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from network import FeedForwardNN



class PPO:

    def __init__(self, env):
        # extract the environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0]

        #initialize the actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()

        #create our variable for the matrix.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        #optimizers for the actor and critic networks
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)




    def learn(self, total_timesteps):
        t_so_far = 0 
        iteration = 0

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            #calculate how many timesteps we collected this batch   
            t_so_far += np.sum(batch_lens)
            iteration += 1

            print(f"\nIteration {iteration}")
            print(f"Total timesteps: {t_so_far}")
            print(f"Average episode length: {np.mean(batch_lens):.2f}")
            print(f"Average episode score: {np.mean(ep_scores):.2f}")

            #calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            print(f"Average value estimate: {V.mean().item():.2f}")

            
            #calculate advantage
            A_k = batch_rtgs - V.detach()


            #normalize the advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            for _ in range(self.n_updates_per_iteration):
                #calc pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k 

                actor_loss = (-torch.min(surr1, surr2)).mean() #actor loss
                critic_loss = nn.MSELoss()(V, batch_rtgs) #critic loss


                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                #for critic network
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

    def _init_hyperparameters(self):

        self.timesteps_per_batch = 4800         # timesteps simulated so far
    
        self.max_timesteps_per_episode = 1600       # max timesteps per episode

        self.gamma = 0.99

        self.n_updates_per_iteration = 5 #epoch num

        self.clip = 0.2

        self.lr = 0.005

    def get_action(self, obs):
        # Ensure obs is a tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            
        # If the environment action space is discrete, use a Categorical distribution.
        if hasattr(self.env.action_space, 'n'):
            logits = self.actor(obs)
            categorical = torch.distributions.Categorical(logits=logits)
            action = categorical.sample()
            log_prob = categorical.log_prob(action)
            return action.item(), log_prob.detach()
        else:
            # Otherwise, for continuous action spaces, use MultivariateNormal.
            mean = self.actor(obs)
            dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.detach().numpy().item(), log_prob.detach()

    def rollout(self):
        # batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

       
        # number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # rewards this episode
            ep_rews = []
            obs = self.env.reset()
            # Unpack observation if necessary (Gym >=0.26)
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # increment timesteps run so far this batch
                t += 1
                # collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                self.env.render()  # Add this line to see the game
                pygame.time.delay(20)
                done = terminated or truncated                # Unpack observation if step() returns a tuple
                if isinstance(obs, tuple):
                    obs = obs[0]
                # collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            # collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
        # convert collected data to tensors and compute rewards-to-go
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    
    def compute_rtgs(self, batch_rews):
        #the rewards-to-go (rtg) per episode per batch to return.
        #the shape will be (num timesteps per episode)
        batch_rtgs = []
        #iterate through each episode backwards to maintain same order
        #in batch_rtgs

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        #convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        
        # If the environment action space is discrete, use Categorical distribution.
        if hasattr(self.env.action_space, 'n'):
            logits = self.actor(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            # Ensure batch_acts is of type long.
            log_probs = dist.log_prob(batch_acts.squeeze().long())
        else:
            # For continuous action spaces, use MultivariateNormal.
            mean = self.actor(batch_obs)
            dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)
        
        return V, log_probs
    
if __name__ == "__main__":
    from flappy_env import FlappyEnv
    env = FlappyEnv()
    model = PPO(env)
    model.learn(10000)
