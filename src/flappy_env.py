import gym
from gym import spaces
import numpy as np
import pygame
from game import Game

class FlappyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FlappyEnv, self).__init__()
        pygame.init()
        self.game = Game()
        # Define the action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        # Define an observation space.
        # For example, we can return a vector of 5 values:
        # [bird y position, bird vertical velocity, distance to next pipe, top height of pipe, bottom y of pipe]
        self.observation_space = spaces.Box(low=0, high=600, shape=(5,), dtype=np.float32)

    def reset(self):
        # Reset the game
        self.game = Game()
        # Return the initial observation
        return self._get_obs()

    def step(self, action):
        # Execute the action: flap if action == 1
        if action == 1:
            self.game.bird.flap()
        # Update the game state
        self.game.update()
        # Check if the game is over
        done = self.game.game_over
        # Reward: 1 per frame survived, -100 on game over
        reward = 1.0 if not done else -100.0
        # Get current observation state
        obs = self._get_obs()
        info = {}
        # Return observation, reward, termination, truncation (False), and info
        return obs, reward, done, False, info

    def _get_obs(self):
            # Get bird position and velocity
        bird_y = self.game.bird.y
        bird_velocity = self.game.bird.velocity

        # Find the nearest pipe ahead of the bird
        nearest_pipe = None
        for pipe in self.game.pipes:
            if pipe.x + pipe.width >= self.game.bird.x:
                nearest_pipe = pipe
                break

        if nearest_pipe is None:
            # If no pipe is found, set default values.
            pipe_dist = 400.0
            pipe_top = 200.0
            pipe_bottom = pipe_top + 150.0
        else:
            pipe_dist = (nearest_pipe.x + nearest_pipe.width) - self.game.bird.x
            pipe_top = nearest_pipe.top_height
            pipe_bottom = nearest_pipe.bottom_y

        return np.array([bird_y, bird_velocity, pipe_dist, pipe_top, pipe_bottom], dtype=np.float32)

    def render(self, mode='human'):
        self.game.draw()
        pygame.display.update()

    def close(self):
        pygame.quit()