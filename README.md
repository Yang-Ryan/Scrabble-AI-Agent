# Scrabble AI Agent

This repository contains the implementation of a Scrabble Agent developed for the NYCU AI Course Final Project. The agent is trained base on Q-Learning with mutiple enhancement.

# Overview

We aim to build a Scrabble agent that learns to play competitively against opponents. The project focuses on implementing four main function:

1. **Experience Replay Buffer**
2. **Target Network (Double DQN Enhancement)**
3. **Multi-Horizon Learning**
4. **Adaptive Components** 

# Prerequisites

* **Python Version**: 3.9+
* **Python Packages**: numpy / matplotlib
* **Operating System**: macOS / Linux / Windows
* **Environment**: Local machine

# Usage

### 1. Training and Evaluating the Agent

Clone the repository
```bash
git clone https://github.com/Yang-Ryan/Scrabble-AI-Agent.git
cd Scrabble-AI-Agent
```

(Use virtual env if you want)

```bash
python3 -m venv scrabble-env
source scrabble-env/bin/activate
```
Install dependencies

```bash
pip install -r requirements.txt
```

Train the Agent

Train the agent via self-play (battle against itself).
By default, the parameters will be:

```bash
python3 main.py self-play \
    --episodes 2000 \
    --learning-rate 0.01 \
    --epsilon 0.3 \
    --buffer-size 10000 \
    --greedy-eval-interval 1 \
    --greedy-eval-games 3 \
    --multi-horizon \
    --save-model
```

Train against a greedy agent:

```bash
python3 main.py train \
    --episodes 2000 \
    --learning-rate 0.01 \
    --epsilon 0.3 \
    --buffer-size 5000 \
    --greedy-eval-interval 1 \
    --greedy-eval-games 3 \
    --multi-horizon \
    --save-model
```

Evaluation

Evaluate the trained agent model with a greedy oponent: (play games and analyze win rate...)

```bash
python3 main.py evaluate 300 \
    --model-path agent_model.json \
    --dictionary dictionary.txt \
    --verbose \
    --save-results \
    --plot
```

#### ---- Configurable Command Line Hyperparameters ----
For Training:
| Argument                 | Default | Description                                     |
| :----------------------- | :-----: | :---------------------------------------------- |
| `--episodes`             |   2000  | Number of training episodes                     |
| `--learning-rate`        |   0.01  | Learning rate for Q-Learning                    |
| `--epsilon`              |   0.3   | Initial exploration rate (ε-greedy policy)      |
| `--buffer-size`          |   5000  | Experience replay buffer size                   |
| `--greedy-eval-interval` |    1    | Evaluate vs greedy every N episodes             |
| `--greedy-eval-games`    |    3    | Number of games per evaluation                  |
| `--multi-horizon`        | `False` | Use multi-step reward estimation                |
| `--save-model`           | `False` | Save model after training                       |


For Evaluating:
| Argument             |      Default     | Description                                 |
| :------------------- | :--------------: | :------------------------------------------ |
| `games` (positional) |     Required     | Number of games to play during evaluation   |
| `--model-path`       |     Required     | Path to the trained model file              |
| `--dictionary`       | `dictionary.txt` | Dictionary file path                        |
| `--verbose`          |      `False`     | Show detailed game-by-game results          |
| `--save-results`     |      `False`     | Save evaluation results to a JSON file      |
| `--plot`             |      `False`     | Generate comprehensive evaluation plots     |


### 2. Play a Scrabble Game

You can also play a Scrabble game against our best trained AI !

```bash
python3 game.py
```

Real Game Demo:

```text

🎮 SCRABBLE vs RL AGENT
==================================================
📋 Board:
   A B C D E F G H I J K L M N O
  ───────────────────────────────
 1│T · · · · · · T · · · · · · T │
 2│· D · · · · · · · · · · · D · │
 3│· · D · · · · · · · · · D · · │
 4│· · · D · · · · · · · D · · · │
 5│· · · · D · · · · · D · · · · │
 6│· · · · · · · · · · · · · · · │
 7│· · · · · · · · · · · · · · · │
 8│T · · · · · · ★ · · · · · · T │
 9│· · · · · · · · · · · · · · · │
10│· · · · · · · · · · · · · · · │
11│· · · · D · · · · · D · · · · │
12│· · · D · · · · · · · D · · · │
13│· · D · · · · · · · · · D · · │
14│· D · · · · · · · · · · · D · │
15│T · · · · · · T · · · · · · T │
  ───────────────────────────────
Legend: DW=Double Word, TW=Triple Word, DL=Double Letter, TL=Triple Letter

📝 Last move: Game started
🎲 Turn: 1
🏆 Scores - You: 0, AI: 0
🤖 AI: Greedy AI (Fallback)
🎯 Tiles left: 86
🎪 Your rack: O N S A X G S
==================================================

🎯 YOUR TURN!

📋 Your moves (showing top 15 by score):
 1. AXONS      -  46 pts - ( 8,F) →
 2. AXONS      -  44 pts - ( 6,H) ↓
 3. SNOGS      -  37 pts - ( 6,H) ↓
 4. AGONS      -  35 pts - ( 8,F) →
 5. SNOGS      -  35 pts - ( 8,F) →
 6. GOSSAN     -  34 pts - ( 8,E) →
 7. GOSSAN     -  34 pts - ( 5,H) ↓
 8. SONGS      -  34 pts - ( 8,F) →
 9. SONGS      -  33 pts - ( 6,H) ↓
10. SAGOS      -  33 pts - ( 8,F) →
11. SNAGS      -  33 pts - ( 6,H) ↓
12. SAGOS      -  32 pts - ( 6,H) ↓
13. SNAGS      -  32 pts - ( 8,F) →
14. AGONS      -  32 pts - ( 6,H) ↓
15. AXON       -  25 pts - ( 8,F) →
... and 45 more moves available

🎮 Commands:
  • Enter move number (1-15)
  • 'more' - show all moves
  • 'pass' - skip turn
  • 'quit' - end game

🎯 Your choice: 
```

# Experiment Results

### Experiments

#### 1. Training Opponent Comparison: Self-Play and AI vs Greedy

* **Greedy-Training** beats Greedy faster and by a larger margin.
* **Self-Play** is slower to surpass Greedy and plateaus lower.

#### 2. Learning Rate Experiments

* α = 0.01: Fast convergence but moderate oscillations.
* α = 0.005: Slow but stable, best overall final performance.
* α = 0.5: Too large, unstable training.

#### 3. Multi-Horizon Learning

* Without multi-horizon: Slow, unstable early learning; low final margin.
* With multi-horizon: Faster early progress, higher and more stable final performance.

#### 4. Experience Replay Size

* Larger buffer (2000): Faster convergence, stable plateau.
* Smaller buffer (1000): Slower convergence, more variance.

### Discussion and Analysis

#### Training Opponent Comparison

* **Greedy‐Training** achieves a 72% win rate and a +36 point gap.
* **Self‐Play** achieves only a 63% win rate with a +28 point gap.
* **Greedy‐Training** exploits the weaknesses of the Greedy agent faster.

#### Learning Rate

* **α = 0.005** provides the best balance: \~67% win rate and +30 point gap.
* **α = 0.01** learns faster early but slightly weaker final performance.
* **α = 0.50** causes instability with high variance in performance.

#### Multi-Horizon Learning

* Speeds up early training by \~100 episodes compared to non-multi-horizon.
* Leads to a +28 point gap compared to \~+15 without multi-horizon.
* Reduces outliers and stabilizes performance across game phases.

#### Experience Replay

* **Large buffer (2000)**: Faster convergence (\~episode 100), higher plateau (+25 to +30 points).
* **Small buffer (1000)**: Slower convergence (\~episode 120), noisier plateau.

### Limitations

* Opponent overfitting to Greedy patterns.
* No modeling of static tile distribution (bag management).
* No evaluation against human players or deeper search engines.

## Project Structure

```bash
Scrabble-AI-Agent/
├── plot/                 # plot results directory
├── scrabble_agent.py     # Q-Learning and Greedy agent implementations
├── move_generator.py     # Generate valid moves based on the board state
├── trainer.py            # Training logic for self-play and evaluation
├── utils.py              # Helper functions: board setup, tile draw, etc.
├── main.py               # Entry point: train and evaluate agents
├── requirements.txt
├── final_model.json      # Our best scrabble model
└── README.md

```

##  Authors

* \[Team 34]

  * \[林鈺凱]
  * \[楊睿軒]
  * \[鄭雯儀]
  * \[千年倫子]
