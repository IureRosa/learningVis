import streamlit as st
import gym
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import Monitor as GymMonitor
from pathlib import Path
import base64
import pybullet as p

# Função para treinar os agentes e plotar os resultados
def train_agents(env_name, max_episodes, ddpg_hyperparams, sac_hyperparams, td3_hyperparams, record_video=True):
    env = gym.make(env_name)
    
    if record_video:
        video_path = Path("videos")
        video_path.mkdir(exist_ok=True)
        env = GymMonitor(env, video_path, force=True)

    env = DummyVecEnv([lambda: env])

    # Criando os agentes
    ddpg_agent = DDPG('MlpPolicy', env, **ddpg_hyperparams)
    sac_agent = SAC('MlpPolicy', env, **sac_hyperparams)
    td3_agent = TD3('MlpPolicy', env, **td3_hyperparams)

    # Listas para armazenar os resultados
    ddpg_rewards = []
    sac_rewards = []
    td3_rewards = []
    ddpg_success_rate = []
    sac_success_rate = []
    td3_success_rate = []

    # Treinamento dos agentes
    for episode in range(max_episodes):
        ddpg_agent.learn(total_timesteps=1000, log_interval=1000)
        sac_agent.learn(total_timesteps=1000, log_interval=1000)
        td3_agent.learn(total_timesteps=1000, log_interval=1000)

        # Calculando a recompensa média do episódio
        ddpg_reward, ddpg_success = evaluate_agent(ddpg_agent, env)
        sac_reward, sac_success = evaluate_agent(sac_agent, env)
        td3_reward, td3_success = evaluate_agent(td3_agent, env)

        # Armazenando as recompensas e taxa de sucesso
        ddpg_rewards.append(ddpg_reward)
        sac_rewards.append(sac_reward)
        td3_rewards.append(td3_reward)
        ddpg_success_rate.append(ddpg_success)
        sac_success_rate.append(sac_success)
        td3_success_rate.append(td3_success)

    # Plotando os resultados
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Reinforcement Learning Results')

    axs[0, 0].plot(ddpg_success_rate, label='DDPG')
    axs[0, 0].plot(sac_success_rate, label='SAC')
    axs[0, 0].plot(td3_success_rate, label='TD3')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Success Rate')
    axs[0, 0].legend()

    axs[0, 1].plot(np.cumsum(ddpg_rewards), label='DDPG')
    axs[0, 1].plot(np.cumsum(sac_rewards), label='SAC')
    axs[0, 1].plot(np.cumsum(td3_rewards), label='TD3')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Cumulative Reward')
    axs[0, 1].legend()

    axs[1, 0].plot(ddpg_rewards, label='DDPG')
    axs[1, 0].plot(sac_rewards, label='SAC')
    axs[1, 0].plot(td3_rewards, label='TD3')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Average Reward')
    axs[1, 0].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Criando um DataFrame com os resultados
    results_df = pd.DataFrame({
        'DDPG Success Rate': ddpg_success_rate,
        'SAC Success Rate': sac_success_rate,
        'TD3 Success Rate': td3_success_rate,
        'DDPG Cumulative Reward': np.cumsum(ddpg_rewards),
        'SAC Cumulative Reward': np.cumsum(sac_rewards),
        'TD3 Cumulative Reward': np.cumsum(td3_rewards),
        'DDPG Average Reward': ddpg_rewards,
        'SAC Average Reward': sac_rewards,
        'TD3 Average Reward': td3_rewards
    })

    # Exibindo a tabela de resultados
    st.subheader('Tabular Results')
    st.dataframe(results_df)

    # Salvando a tabela de resultados em um arquivo CSV
    save_results_as_csv(results_df)

    # Salvando o gráfico como imagem
    save_plot_as_png(fig)

    # Salvando o vídeo do treinamento
    if record_video:
        save_video_as_mp4(video_path)


def save_results_as_csv(results_df):
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parent_directory = "results"
    sub_directory = f"{script_name}_{timestamp}"
    directory = os.path.join(parent_directory, sub_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f"{script_name}_{timestamp}.csv")
    results_df.to_csv(filename, index=False)
    print(f"Results saved as {filename}")


def save_plot_as_png(figure):
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parent_directory = "results"
    sub_directory = f"{script_name}_{timestamp}"
    directory = os.path.join(parent_directory, sub_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f"{script_name}_{timestamp}.png")
    figure.savefig(filename, format='png')
    plt.close(figure)
    print(f"Plot saved as {filename}")


def save_video_as_mp4(video_path):
    video_files = list(video_path.glob("*.mp4"))
    if len(video_files) > 0:
        video_file = video_files[0]
        with open(video_file, "rb") as file:
            video_bytes = file.read()
            b64_video = base64.b64encode(video_bytes).decode()
            st.video(f"data:video/mp4;base64,{b64_video}")
        print(f"Video saved as {video_file}")


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
    st.title("Continuous Action Space Results")

    st.sidebar.title("Hyperparameters")
    env_names = ['Pendulum-v1', 'BipedalWalker-v3', 'HalfCheetah-v3']
    env_name = st.sidebar.selectbox("Choose environment", env_names)

    max_episodes = st.sidebar.number_input("Max Episodes", value=100)

    ddpg_hyperparams = {
        'policy_kwargs': dict(net_arch=[256, 256]),
        'learning_starts': 100,
        'buffer_size': 10000,
        'batch_size': 64,
        'learning_rate': st.sidebar.number_input("DDPG Learning Rate", value=1e-3),
        'gamma': st.sidebar.number_input("DDPG Gamma", value=0.99),
        'tau': st.sidebar.number_input("DDPG Tau", value=0.001)
    }

    sac_hyperparams = {
        'policy_kwargs': dict(net_arch=[256, 256]),
        'learning_starts': 100,
        'buffer_size': 10000,
        'batch_size': 64,
        'learning_rate': st.sidebar.number_input("SAC Learning Rate", value=1e-3),
        'gamma': st.sidebar.number_input("SAC Gamma", value=0.99),
        'tau': st.sidebar.number_input("SAC Tau", value=0.005),
        'target_entropy': 'auto'
    }

    td3_hyperparams = {
        'policy_kwargs': dict(net_arch=[256, 256]),
        'learning_starts': 100,
        'buffer_size': 10000,
        'batch_size': 64,
        'learning_rate': st.sidebar.number_input("TD3 Learning Rate", value=1e-3),
        'gamma': st.sidebar.number_input("TD3 Gamma", value=0.99),
        'tau': st.sidebar.number_input("TD3 Tau", value=0.005),
        'target_policy_noise': st.sidebar.number_input("TD3 Target Policy Noise", value=0.2),
        'target_noise_clip': st.sidebar.number_input("TD3 Target Noise Clip", value=0.5)
    }

    record_video = st.sidebar.checkbox("Record Training Video")

    if st.sidebar.button("Train"):
        train_agents(env_name, max_episodes, ddpg_hyperparams, sac_hyperparams, td3_hyperparams, record_video)


if __name__ == '__main__':
    main()
