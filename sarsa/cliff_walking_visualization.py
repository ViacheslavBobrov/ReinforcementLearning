import gym
from sarsa_agents import ExpectedSarsaAgent

env = gym.make('CliffWalking-v0')
agent = ExpectedSarsaAgent(env)
agent.play_n_steps(50000)
agent.play_episode(render=True)
