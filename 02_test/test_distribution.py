import torch
import numpy as np
from environment import TetrisEnv
from model import DQN

import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model_path, episodes=500):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TetrisEnv()

    model = DQN(env.state_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_scores = []
    total_lines = []

    for ep in range(episodes):
        env.reset()
        env.spawn_piece()
        
        while not env.game_over:
            actions = env.get_possible_actions()
            if not actions:
                break

            states = []
            valid_actions = []
            for action in actions:
                s = env.get_state_for_action(action)
                if s is not None:
                    states.append(s)
                    valid_actions.append(action)

            if not valid_actions:
                break

            with torch.no_grad():
                x = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                q_values = model(x).squeeze(1)
                best_idx = torch.argmax(q_values).item()

            env.step(valid_actions[best_idx])

        print(f"Test Episode {ep + 1:2} | Score: {env.score:6} | Lines Cleared: {env.lines_cleared}")
        total_scores.append(env.score)
        total_lines.append(env.lines_cleared)

    print("-" * 40)
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Average Lines: {np.mean(total_lines):.2f}")

    # plot distribution of scores and lines cleared
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(total_scores, bins=20, kde=True)
    sns.lineplot(x=[np.mean(total_scores)], y=[0], marker="o", color="red", label=f"Mean: {np.mean(total_scores):.2f}")
    sns.lineplot(x=[np.median(total_scores)], y=[0], marker="o", color="green", label=f"Median: {np.median(total_scores):.2f}")
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(total_lines, bins=20, kde=True)
    sns.lineplot(x=[np.mean(total_lines)], y=[0], marker="o", color="red", label=f"Mean: {np.mean(total_lines):.2f}")
    sns.lineplot(x=[np.median(total_lines)], y=[0], marker="o", color="green", label=f"Median: {np.median(total_lines):.2f}")
    plt.title("Distribution of Lines Cleared")
    plt.xlabel("Lines Cleared")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    # save the plot
    plt.savefig("score_line_distribution.png")
    plt.show()


if __name__ == "__main__":
    evaluate("tetris_dqn.pt")