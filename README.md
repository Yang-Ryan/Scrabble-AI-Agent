## Installation
I recommend using virtual env

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

To train our model, you can add many command line args: 
1. train with greedy agent (RL vs Greedy)
2. train with yourself (RL vs RL) <- this method will provide a three-game evaluation with greedy agent at every episode  
3. play game!

```bash
python main.py train \
    --episodes 1000 \
    --learning-rate 0.02 \
    --epsilon 0.4 \
    --gamma 0.95 \
    --eval-interval 25 \
    --multi-horizon \
    --save-model \

python main.py self-play \
    --episodes 5000 \
    --learning-rate 0.015 \
    --epsilon 0.4 \
    --buffer-size 10000 \
    --greedy-eval-interval 5 \
    --greedy-eval-games 5 \
    --multi-horizon \
    --save-model

python3 main.py play

(scrabble-env) $ python3 main.py self-play --episodes 2000 --greedy-eval-games 3 --save-model
(scrabble-env) $ python3 main.py train --episodes 500 --save-model

```

---

## Explain For Each File

    Scrabble-AI-Agent/
    ├── main.py # train and evaluate
    ├── scrabble_game.py # game logic...
    ├── rl_agent.py # Q-Learning Agent
    ├── baseline_agent.py # baseline agents to compete with（Random / Greedy / Heuristic / Adaptive）
    ├── evaluation.py # generate games vs baseline agents and plot
    ├── dictionary.txt
    ├── rl_model_.json # trained Q-table model
    ├── evaluation_results_.json
    ├── evaluation_analysis_.json
    ├── evaluation_plots_.png # plot
    └── README.md