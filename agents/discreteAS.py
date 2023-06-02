import streamlit as st
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN, PPO

# Função para treinar os agentes e plotar os resultados
def train_agents(env_name, max_episodes, dqn_hyperparams, ppo_hyperparams):
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    # Criando o agente DQN
    dqn_agent = DQN('MlpPolicy', env, **dqn_hyperparams)

    # Criando o agente PPO
    ppo_agent = PPO('MlpPolicy', env, **ppo_hyperparams)

    # Listas para armazenar os resultados
    dqn_rewards = []
    dqn_success_rate = []
    dqn_avg_rewards = []
    ppo_rewards = []
    ppo_success_rate = []
    ppo_avg_rewards = []

    # Treinamento dos agentes
    for episode in range(max_episodes):
        # Treinamento do agente DQN
        dqn_agent.learn(total_timesteps=1000, log_interval=1000)

        # Calculando a recompensa média e taxa de sucesso do episódio para DQN
        dqn_reward, dqn_success = evaluate_agent(dqn_agent, env)

        # Armazenando as recompensas, taxa de sucesso e recompensa média para DQN
        dqn_rewards.append(dqn_reward)
        dqn_success_rate.append(dqn_success)
        dqn_avg_rewards.append(np.mean(dqn_rewards))

        # Treinamento do agente PPO
        ppo_agent.learn(total_timesteps=1000, log_interval=1000)

        # Calculando a recompensa média e taxa de sucesso do episódio para PPO
        ppo_reward, ppo_success = evaluate_agent(ppo_agent, env)

        # Armazenando as recompensas, taxa de sucesso e recompensa média para PPO
        ppo_rewards.append(ppo_reward)
        ppo_success_rate.append(ppo_success)
        ppo_avg_rewards.append(np.mean(ppo_rewards))

    # Plotando os resultados
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Reinforcement Learning Results')

    axs[0].plot(dqn_success_rate, label='DQN')
    axs[0].plot(ppo_success_rate, label='PPO')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Success Rate')
    axs[0].legend()

    axs[1].plot(np.cumsum(dqn_rewards), label='DQN')
    axs[1].plot(np.cumsum(ppo_rewards), label='PPO')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Cumulative Reward')
    axs[1].legend()

    axs[2].plot(dqn_avg_rewards, label='DQN')
    axs[2].plot(ppo_avg_rewards, label='PPO')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Reward')
    axs[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Função para avaliar um agente
def evaluate_agent(agent, env):
    episode_reward = 0
    episode_success = 0
    done = False
    obs = env.reset()
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if 'success' in info and info['success']:
            episode_success = 1
    return episode_reward, episode_success

# Interface do Streamlit
def main():
    st.title("Reinforcement Learning Algorithms")

    st.sidebar.title("Hyperparameters")
    env_names = ['MountainCar-v0', 'CartPole-v1', 'LunarLander-v2', 'FrozenLake-v1', 'FrozenLake-v8']
    env_name = st.sidebar.selectbox("Choose environment", env_names)

    max_episodes = st.sidebar.number_input("Max Episodes", value=100)

    dqn_hyperparams = {
        'learning_starts': 100,
        'buffer_size': 10000,
        'learning_rate': st.sidebar.number_input("DQN Learning Rate", value=1e-3),
        'gamma': st.sidebar.number_input("DQN Gamma", value=0.99),
        'exploration_fraction': st.sidebar.number_input("DQN Exploration Fraction", value=0.1),
        'exploration_final_eps': st.sidebar.number_input("DQN Exploration Final Epsilon", value=0.02),
        'target_update_interval': st.sidebar.number_input("DQN Target Update Interval", value=1000),
    }

    ppo_hyperparams = {
        'learning_rate': st.sidebar.number_input("PPO Learning Rate", value=3e-4),
        'gamma': st.sidebar.number_input("PPO Gamma", value=0.99),
        'gae_lambda': st.sidebar.number_input("PPO GAE Lambda", value=0.95),
        'clip_range': st.sidebar.number_input("PPO Clip Range", value=0.2),
        'ent_coef': st.sidebar.number_input("PPO Entropy Coefficient", value=0.01),
        'vf_coef': st.sidebar.number_input("PPO Value Function Coefficient", value=0.5),
        'max_grad_norm': st.sidebar.number_input("PPO Max Gradient Norm", value=0.5),
        'n_epochs': st.sidebar.number_input("PPO Number of Epochs", value=4),
        'batch_size': st.sidebar.number_input("PPO Batch Size", value=32),
    }

    if st.sidebar.button("Train"):
        train_agents(env_name, max_episodes, dqn_hyperparams, ppo_hyperparams)

if __name__ == '__main__':
    main()
