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

EPISODES = 2000
BATCH_SIZE = 512
GAMMA = 0.98
LR = 3e-4
MEMORY_SIZE = 30000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9985
TARGET_UPDATE = 10
PLOT_UPDATE_INTERVAL = 100
GRAD_CLIP_NORM = 10.0
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
RANDOM_START_PROB = 0.40

torch.set_num_threads(TORCH_NUM_THREADS)

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


def randomize_training_board(
    self,
    min_fill=0.30,
    max_fill=0.60,
    min_hole_rate=0.05,
    max_hole_rate=0.20,
    max_attempts=100,
):
    for _ in range(max_attempts):
        board = np.zeros((self.rows, self.cols), dtype=np.float32)

        fill_rate = random.uniform(min_fill, max_fill)
        hole_rate = random.uniform(min_hole_rate, max_hole_rate)

        target_filled_cells = int(fill_rate * self.rows * self.cols)
        target_stack_area = int(target_filled_cells / max(1e-6, 1.0 - hole_rate))

        avg_height = target_stack_area / self.cols

        # Generate a varied but stack-like surface.
        heights = np.random.normal(
            loc=avg_height,
            scale=max(1.0, avg_height * 0.35),
            size=self.cols,
        )

        heights = np.clip(np.round(heights), 1, self.rows - 2).astype(int)

        # Build solid stacks first.
        for c, h in enumerate(heights):
            board[self.rows - h:self.rows, c] = 1.0

        # Carve holes from inside the stacks.
        candidates = []
        for c, h in enumerate(heights):
            top_row = self.rows - h

            # Exclude the top cell so the column height remains meaningful.
            for r in range(top_row + 1, self.rows):
                candidates.append((r, c))

        random.shuffle(candidates)

        num_holes = int(hole_rate * sum(heights))
        for r, c in candidates[:num_holes]:
            board[r, c] = 0.0

        # A post-clear Tetris board should not already contain full rows.
        # Break any accidental full rows.
        for r in range(self.rows):
            if np.all(board[r] > 0):
                c = random.randrange(self.cols)
                board[r, c] = 0.0

        self.board = board
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.current_piece = self._random_piece()
        self.next_piece = self._random_piece()
        self.piece_active = False

        # Require at least one real possible action.
        valid_actions = [
            a for a in self.get_possible_actions()
            if self.get_state_for_action(a) is not None
        ]

        if valid_actions:
            return True

    self.reset()
    return False


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
    criterion = nn.SmoothL1Loss() # smooth L1 loss for regression of q-values

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    # metrics
    scores_history = []
    lines_history = []
    epsilon_history = []

    for episode in range(1, EPISODES + 1):

        if random.random() < RANDOM_START_PROB:
            env.randomize_training_board(
                min_fill=0.30,
                max_fill=0.60,
                min_hole_rate=0.05,
                max_hole_rate=0.20,
            )
        else:
            env.reset()

        done = False
        info = {"score": 0, "lines_cleared": 0}
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
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
