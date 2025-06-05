import gymnasium as gym

def simple_policy(obs):
    """
    Choose an action based on the pole's angle.
    Push right (1) if angle > 0, else push left (0).
    """
    return int(obs[2] > 0)

def run_cartpole(env, total_timesteps=2500):
    obs, info = env.reset(seed=42)
    episode_count = 0
    episode_steps = 0
    cumulative_steps = 0

    for t in range(total_timesteps):
        env.render()

        action = simple_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1

        if terminated or truncated:
            print(f"Episode {episode_count + 1} lasted {episode_steps} steps")
            cumulative_steps += episode_steps
            episode_steps = 0
            episode_count += 1
            obs, info = env.reset()

    env.close()
    if episode_count > 0:
        print(f"\nAverage steps per episode: {cumulative_steps // episode_count}")
    else:
        print("\nNo completed episodes.")

if __name__ == "__main__":
    environment = gym.make("CartPole-v1", render_mode="human")
    run_cartpole(environment)
