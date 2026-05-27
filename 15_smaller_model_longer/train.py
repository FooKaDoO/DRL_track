import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

import matplotlib.pyplot as plt

import os
import csv
from collections import deque

from environment import TetrisEnv
from model import DQN

EPISODES = 20000
BATCH_SIZE = 512
GAMMA = 0.98
LR = 1e-3
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.999
TARGET_UPDATE = 10
PLOT_UPDATE_INTERVAL = 100
GRAD_CLIP_NORM = 10.0
LOAD_MODEL_PATH = "tetris_dqn.pt"
LOAD_EXISTING_MODEL = True
RESUME_EPSILON = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def should_save_checkpoint(episode):

    remaining = EPISODES - episode

    if remaining > EPISODES // 2:
        return episode % (EPISODES // 10) == 0
    if remaining > EPISODES // 5:
        return episode % (EPISODES // 20) == 0
    if remaining > (3 * EPISODES) // 50:
        return episode % (EPISODES // 50) == 0
    return episode % (EPISODES // 100) == 0


def save_metrics_and_plots(scores, lines, epsilons, filename_prefix="training"):

    """Saves training data to CSV and generates a plot PNG."""
    
    # 1. Save data to CSV
    csv_filename = f"{filename_prefix}_metrics.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score", "Lines_Cleared", "Epsilon"])
        for i in range(len(scores)):
            writer.writerow([i + 1, scores[i], lines[i], epsilons[i]])

    # 2. Generate and save Plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Scores
    plt.subplot(1, 3, 1)
    plt.plot(scores, color='blue', alpha=0.3, label="Score")
    if len(scores) >= 50:
        # Calculate 50-episode moving average
        moving_avg = np.convolve(scores, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(scores)), moving_avg, color='blue', label="50-ep SMA")
    plt.title("Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    # Plot 2: Lines Cleared
    plt.subplot(1, 3, 2)
    plt.plot(lines, color='green', alpha=0.5, label="Lines Cleared")
    if len(lines) >= 50:
        lines_avg = np.convolve(lines, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(lines)), lines_avg, color='darkgreen', label="50-ep SMA")
    plt.title("Lines Cleared per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Lines")
    plt.legend()

    # Plot 3: Epsilon Decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilons, color='orange', label="Epsilon")
    plt.title("Exploration Rate (Epsilon)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.tight_layout()
    plot_filename = f"{filename_prefix}_plot.png"
    plt.savefig(plot_filename)
    plt.close()


def train():

    env = TetrisEnv()
    model = DQN(env.state_size).to(device)

    if LOAD_EXISTING_MODEL and os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading model from '{LOAD_MODEL_PATH}'...")
        state_dict = torch.load(LOAD_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded existing model. Continuing training.")
    else:
        print("No existing model loaded. Training from scratch.")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"State size: {env.state_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    target_model = DQN(env.state_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    memory = deque(maxlen=MEMORY_SIZE)

    if LOAD_EXISTING_MODEL and os.path.exists(LOAD_MODEL_PATH):
        epsilon = RESUME_EPSILON
    else:
        epsilon = EPSILON_START

    # metrics
    scores_history = []
    lines_history = []
    epsilon_history = []

    for episode in range(1, EPISODES + 1):
        
        env.reset()
        env.spawn_piece()
        done = False
        
        while not done:

            actions = env.get_possible_actions()

            if not actions:
                break
                
            # evaluate the states for the current piece
            next_states = []
            valid_actions = []

            for action in actions:

                state = env.get_state_for_action(action)

                if state is not None:
                    next_states.append(state)
                    valid_actions.append(action)
                    
            if not valid_actions:
                break
                
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            
            # epsilon-greedy action selection
            # with probability epsilon select a random action, otherwise select the action with the highest Q-value
            if random.random() < epsilon:
                best_idx = random.randint(0, len(valid_actions) - 1)
            else:
                with torch.no_grad():

                    q_values = model(next_states_tensor).squeeze(1)
                    best_idx = torch.argmax(q_values).item()

            best_action = valid_actions[best_idx]
            chosen_state = next_states[best_idx]

            _, reward, done, info = env.step(best_action)
            
            # see next piece's possible states for target Q-value
            next_next_states = []
            if not done:
                next_actions = env.get_possible_actions()

                for a in next_actions:
                    s = env.get_state_for_action(a)
                    if s is not None:
                        next_next_states.append(s)

            # store transition in memory
            memory.append((chosen_state, reward, done, next_next_states))

            # learn from memory if we have enough samples

            if len(memory) >= BATCH_SIZE:

                batch = random.sample(memory, BATCH_SIZE)
                
                b_states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=device)
                b_rewards = torch.tensor([t[1] for t in batch], dtype=torch.float32, device=device)
                b_dones = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=device)
                
                # q targets
                b_targets = b_rewards.clone()

                next_state_batches = []
                next_state_batch_indices = []
                next_state_counts = []

                for i, transition in enumerate(batch):

                    if not b_dones[i] and len(transition[3]) > 0:

                        next_state_batches.append(np.array(transition[3], dtype=np.float32))
                        next_state_batch_indices.append(i)
                        next_state_counts.append(len(transition[3]))

                if next_state_batches:

                    nn_states = torch.tensor(
                        np.concatenate(next_state_batches, axis=0),
                        dtype=torch.float32,
                        device=device,
                    )

                    with torch.no_grad():
                        # Online model chooses the best next state
                        online_next_q = model(nn_states).squeeze(1)

                        # Target model evaluates that chosen next state
                        target_next_q = target_model(nn_states).squeeze(1)

                    offset = 0
                    for batch_index, count in zip(next_state_batch_indices, next_state_counts):
                        q_slice_online = online_next_q[offset:offset + count]
                        q_slice_target = target_next_q[offset:offset + count]

                        best_next_idx = torch.argmax(q_slice_online)
                        max_q_next = q_slice_target[best_next_idx]

                        b_targets[batch_index] += GAMMA * max_q_next
                        offset += count
                            
                b_targets = b_targets.unsqueeze(1)
                
                # current q values
                current_q = model(b_states)
                
                loss = criterion(current_q, b_targets)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        # Update metrics
        scores_history.append(info['score'])
        lines_history.append(info['lines_cleared'])
        epsilon_history.append(epsilon)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 10 == 0:
            print(f"Episode: {episode:4} | Score: {info['score']:6.1f} | Lines: {info['lines_cleared']:4} | Epsilon: {epsilon:.3f}")

        if episode % 100 == 0:
            torch.save(model.state_dict(), "tetris_dqn.pt")

        if should_save_checkpoint(episode):
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"tetris_dqn_ep{episode:05d}.pt"),
            )

        # Update plots periodically
        if episode % PLOT_UPDATE_INTERVAL == 0:
            save_metrics_and_plots(scores_history, lines_history, epsilon_history)

    # Final save at the end of training
    torch.save(model.state_dict(), "tetris_dqn.pt")
    save_metrics_and_plots(scores_history, lines_history, epsilon_history)
    print("Training Complete. Model saved to 'tetris_dqn.pt'. Data saved to 'training_metrics.csv' and 'training_plot.png'.")

if __name__ == "__main__":

    train()
