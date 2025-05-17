#training loop

def train_dqn(agent, env, num_episodes=1000, batch_size=32, max_steps=10000, log_every=10):
    episode_rewards = []

    for episode in trange(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store(state, action, reward, next_state, done)
            agent.learn(batch_size)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % log_every == 0:
            avg_reward = sum(episode_rewards[-log_every:]) / log_every
            print(f"Episode {episode+1}, Avg Reward (last {log_every}): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return episode_rewards