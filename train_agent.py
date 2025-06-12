from stable_baselines3 import PPO
from rl_env import DatasetEnv

def train_rl_agent(df, problem_type):
    env = DatasetEnv(df, problem_type)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)
    return model
