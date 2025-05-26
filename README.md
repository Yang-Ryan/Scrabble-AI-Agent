## Installation

```bash
# Clone the repository
$ git clone https://github.com/Yang-Ryan/Scrabble-AI-Agent.git
$ cd Scrabble-AI-Agent

# virtual environment
$ python3 -m venv scrabble-env
$ source scrabble-env/bin/activate

# Install required packages
(scrabble-env) $ pip install -r requirements.txt
```

---

## Training the RL Agent

To train the agent using Q-Learning against a baseline (default: Greedy agent):

```bash
(scrabble-env) $ python3 main.py train --episodes 500 --save-model
```

- You can change number of episodes with `--episodes`.
- `--save-model` will export the trained Q-table into a `.json` file for future evaluation.

---

## Evaluation & Visualization

Once trained, evaluate the agent against four baselines: Random, Greedy, Heuristic, Adaptive:

```bash
(scrabble-env) $ python3 main.py evaluate --model-path rl_model_2025XXXX.json --eval-games 100
```

Replace `rl_model_2025XXXX.json` with your actual model path.

This will:
- Run 100 evaluation games per baseline.
- Generate win rate stats, score distributions, and average move time.
- Produce 3 output files:
  - `evaluation_results_<timestamp>.json`
  - `evaluation_analysis_<timestamp>.json`
  - `evaluation_plots_<timestamp>.png`
---