import streamlit as st
import gym
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Função para executar o algoritmo Q-learning
def q_learning(env, alpha, gamma, epsilon, max_episodes):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    success_rate = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_success = 0
        done = False

        while not done:
            # Escolha da ação usando a política epsilon-greedy
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)

            # Atualização da função Q
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            episode_reward += reward

            if done and reward == 1:
                episode_success = 1

        rewards.append(episode_reward)
        success_rate.append(episode_success)

    return Q, rewards, success_rate


# Função para executar o algoritmo SARSA
def sarsa(env, alpha, gamma, epsilon, max_episodes):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    success_rate = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_success = 0
        done = False

        # Escolha da ação inicial usando a política epsilon-greedy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        while not done:
            next_state, reward, done, _ = env.step(action)

            # Escolha da próxima ação usando a política epsilon-greedy
            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # Atualização da função Q
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
            episode_reward += reward

            if done and reward == 1:
                episode_success = 1

        rewards.append(episode_reward)
        success_rate.append(episode_success)

    return Q, rewards, success_rate


# Função para plotar os resultados
def plot_results(q_rewards, q_success_rate, sarsa_rewards, sarsa_success_rate):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Reinforcement Learning Results')

    axs[0].plot(np.cumsum(q_rewards), label='Q-learning')
    axs[0].plot(np.cumsum(sarsa_rewards), label='SARSA')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Cumulative Reward')
    axs[0].legend()

    axs[1].plot(q_success_rate, label='Q-learning')
    axs[1].plot(sarsa_success_rate, label='SARSA')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Success Rate')
    axs[1].legend()

    axs[2].plot(np.convolve(q_rewards, np.ones(100) / 100, mode='valid'), label='Q-learning')
    axs[2].plot(np.convolve(sarsa_rewards, np.ones(100) / 100, mode='valid'), label='SARSA')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Reward')
    axs[2].legend()

    plt.tight_layout()
    st.pyplot(fig)
    save_plot_as_png(fig)

    # Tabela com resultados numéricos
    results_data = {
        'Algorithm': ['Q-learning', 'SARSA'],
        'Total Rewards': [sum(q_rewards), sum(sarsa_rewards)],
        'Success Rate': [sum(q_success_rate) / len(q_success_rate), sum(sarsa_success_rate) / len(sarsa_success_rate)]
    }
    results_df = pd.DataFrame(results_data)
    st.write(results_df)


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


# Interface do Streamlit
def main():
    st.title("Reinforcement Learning Algorithms")

    st.sidebar.title("Hyperparameters")
    env_names = ['FrozenLake-v1', 'FrozenLake-v8']
    env_name = st.sidebar.selectbox("Choose environment", env_names)

    alpha_q = st.sidebar.number_input("Q-learning Learning Rate", value=0.1)
    gamma_q = st.sidebar.number_input("Q-learning Gamma", value=0.9)
    epsilon_q = st.sidebar.number_input("Q-learning Epsilon", value=0.5)

    alpha_sarsa = st.sidebar.number_input("SARSA Learning Rate", value=0.1)
    gamma_sarsa = st.sidebar.number_input("SARSA Gamma", value=0.9)
    epsilon_sarsa = st.sidebar.number_input("SARSA Epsilon", value=0.5)

    max_episodes = st.sidebar.number_input("Max Episodes", value=1000)

    if st.sidebar.button("Train"):
        if env_name == 'FrozenLake-v1':
            env = gym.make('FrozenLake-v1')
        elif env_name == 'FrozenLake-v8':
            env = gym.make('FrozenLake8x8-v1')

        # Execução dos algoritmos
        q_learning_Q, q_rewards, q_success_rate = q_learning(env, alpha_q, gamma_q, epsilon_q, max_episodes)
        sarsa_Q, sarsa_rewards, sarsa_success_rate = sarsa(env, alpha_sarsa, gamma_sarsa, epsilon_sarsa, max_episodes)

        # Plot dos resultados
        plot_results(q_rewards, q_success_rate, sarsa_rewards, sarsa_success_rate)


if __name__ == '__main__':
    main()
