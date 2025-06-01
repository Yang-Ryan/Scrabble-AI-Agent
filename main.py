"""
Enhanced Main entry point for Scrabble RL Agent
Now supports regular training, self-play training, and Human vs AI gameplay
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Import all necessary components
from scrabble_agent import AdaptiveScrabbleQLearner, GreedyAgent
from trainer import SelfPlayTrainer, QuackleTrainer
from utils import save_game_data

def train_agent(args):
    """Train agent against greedy opponent"""
    print("🎯 REGULAR TRAINING MODE")
    print("=" * 40)
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Opponent: {args.opponent}")
    print(f"Dictionary: {args.dictionary}")
    print()
    
    # Create agent
    agent = AdaptiveScrabbleQLearner(
        num_features=8,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        gamma=args.gamma,
        buffer_size=2000,
        batch_size=32,
        target_update_frequency=100,
        use_multi_horizon=args.multi_horizon  
    )
    
    # Create trainer
    trainer = SelfPlayTrainer(args.dictionary)
    
    # Train agent against greedy opponent
    trained_agent = trainer._train_vs_greedy(
        agent=agent,
        num_episodes=args.episodes,
        evaluation_interval=args.eval_interval,
        save_interval=args.save_interval,
        verbose=True
    )
    
    # Save trained model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"rl_model_{timestamp}_ep{args.episodes}.json"
        trained_agent.save_model(model_path)
        print(f"\nTrained model saved: {model_path}")
        
        # Save training data
        training_data_path = f"training_data_{timestamp}.json"
        trainer.save_training_data(training_data_path)
        print(f"Training data saved: {training_data_path}")
    
    # Final performance summary
    summary = trainer.get_training_summary()
    if summary.get('final_performance'):
        perf = summary['final_performance']
        print(f"\nFinal Training Performance:")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print(f"Average Score: {perf['avg_score']:.1f}")
        print(f"Average Score Gap: {perf['avg_score_gap']:+.1f}")
    
    return trained_agent


def train_self_play_agent(args):
    """Train agent using self-play with greedy evaluation"""
    if SelfPlayTrainer is None:
        print("❌ Self-play trainer not available. Please check self_play_trainer.py")
        return None
    
    print("🤖 SELF-PLAY TRAINING MODE")
    print("=" * 50)
    print(f"Episodes: {args.episodes} (RL vs RL)")
    print(f"Greedy Evaluation: Every {args.greedy_eval_interval} episode(s)")
    print(f"Games per Evaluation: {args.greedy_eval_games}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Buffer Size: {args.buffer_size}")
    print()
    
    # Create agent
    agent = AdaptiveScrabbleQLearner(
        num_features=8,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=32,
        target_update_frequency=100,
        use_multi_horizon=args.multi_horizon  
    )
    
    # Create self-play trainer
    trainer = SelfPlayTrainer(args.dictionary)
    
    # Train with self-play
    trained_agent = trainer.train_self_play(
        agent=agent,
        num_episodes=args.episodes,
        greedy_eval_interval=args.greedy_eval_interval,
        greedy_eval_games=args.greedy_eval_games,
        verbose=True
    )
    
    # Save trained model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"self_play_model_{timestamp}_ep{args.episodes}.json"
        trained_agent.save_model(model_path)
        print(f"\nTrained model saved: {model_path}")
        
        # Save training data
        training_data_path = f"self_play_data_{timestamp}.json"
        trainer.save_training_data(training_data_path)
        print(f"Training data saved: {training_data_path}")
    
    return trained_agent

def train_quackle_agent(args):
    """Train agent against Quackle opponent using QuackleTrainer"""
    print("🦆 QUACKLE TRAINING MODE")
    print("=" * 50)
    print(f"Episodes: {args.episodes} (RL vs Quackle)")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Gamma: {args.gamma}")
    print(f"Multi-horizon: {args.multi_horizon}")
    print(f"Dictionary: {args.dictionary}")
    print()
    
    # Create agent
    agent = AdaptiveScrabbleQLearner(
        num_features=8,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        gamma=args.gamma,
        buffer_size=getattr(args, 'buffer_size', 2000),
        batch_size=32,
        target_update_frequency=100,
        use_multi_horizon=args.multi_horizon  
    )
    
    # Create Quackle trainer
    quackle_trainer = QuackleTrainer(args.dictionary)
    
    # Train agent against Quackle opponent
    trained_agent = quackle_trainer.train_vs_quackle(
        agent=agent,
        num_episodes=args.episodes,
        evaluation_interval=args.eval_interval,
        save_interval=getattr(args, 'save_interval', 250),
        verbose=True
    )
    
    # Save trained model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"quackle_model_{timestamp}_ep{args.episodes}.json"
        trained_agent.save_model(model_path)
        print(f"\nTrained model saved: {model_path}")
        
        # Save training data
        training_data_path = f"quackle_training_data_{timestamp}.json"
        quackle_trainer.save_training_data(training_data_path)
        print(f"Training data saved: {training_data_path}")
    
    # Final performance summary
    summary = quackle_trainer.get_training_summary()
    if summary.get('final_performance'):
        perf = summary['final_performance']
        print(f"\nFinal Training Performance vs Quackle:")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print(f"Average Score: {perf['avg_score']:.1f}")
        print(f"Average Score Gap: {perf['avg_score_gap']:+.1f}")
    
    return trained_agent

def play_vs_human(args):
    """Launch Human vs AI game interface"""
    print("🎮 LAUNCHING HUMAN VS AI GAME")
    print("=" * 40)

    try:
        game = ScrabbleGameGUI(dictionary_path=args.dictionary)

        # 自動載入 model
        if getattr(args, 'model_path', None):
            if Path(args.model_path).exists():
                agent = AdaptiveScrabbleQLearner()
                agent.load_model(args.model_path)
                game.ai_agent = agent
                game.ai_status_label.config(text="AI Loaded ✅", fg='#27ae60')
                game.log_message(f"🤖 Auto-loaded: {Path(args.model_path).name}")
            else:
                print(f"⚠️ Model not found: {args.model_path}")
        
        game.run()

    except Exception as e:
        print(f"❌ Error launching game: {e}")
        import traceback
        traceback.print_exc()

def evaluate_agent(args):
    """Evaluate a trained agent against greedy opponent"""
    print("📊 AGENT EVALUATION")
    print("=" * 40)
    print(f"Model: {args.model_path}")
    print(f"Games: {args.eval_games}")
    print(f"Dictionary: {args.dictionary}")
    print()
    
    # Load trained agent
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return None
    
    agent = AdaptiveScrabbleQLearner()
    agent.load_model(args.model_path)
    
    print(f"Loaded agent with {agent.training_episodes} training episodes")
    
    # Create trainer for evaluation
    trainer = EnhancedScrabbleTrainer(args.dictionary)
    
    # Run evaluation
    results = trainer._evaluate_agent(agent, num_games=args.eval_games)
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"evaluation_results_{timestamp}.json"
    save_game_data(results, results_path)
    print(f"\nResults saved: {results_path}")
    
    # Print summary
    print_evaluation_summary(results)
    
    return results

def print_evaluation_summary(results):
    """Print formatted evaluation summary"""
    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY")
    print("="*50)
    
    print(f"\nPerformance vs Greedy Opponent:")
    print(f"  Win Rate: {results.get('win_rate', 0):>6.1%}")
    print(f"  Avg Score: {results.get('avg_score', 0):>5.1f}")
    print(f"  Score Gap: {results.get('avg_score_gap', 0):>+5.1f}")
    print(f"  Total Games: {results.get('games_played', 0):>4d}")
    
    # Performance assessment
    win_rate = results.get('win_rate', 0)
    if win_rate > 0.7:
        print(f"\n🚀 EXCELLENT - Dominates greedy opponent!")
    elif win_rate > 0.6:
        print(f"\n✅ GOOD - Consistently beats greedy opponent")
    elif win_rate > 0.5:
        print(f"\n📊 FAIR - Competitive with greedy opponent")
    elif win_rate > 0.4:
        print(f"\n⚠️ BELOW AVERAGE - Struggles against greedy")
    else:
        print(f"\n❌ POOR - Loses consistently to greedy")
    
    print("="*50)

def main():
    """Main entry point with enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='🎮 Scrabble RL Agent - Training, Analysis & Human Play')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Regular training command
    train_parser = subparsers.add_parser('train-greedy', help='Train agent vs greedy opponent')
    train_parser.add_argument('--episodes', type=int, default=500,
                             help='Number of training episodes (default: 500)')
    train_parser.add_argument('--learning-rate', type=float, default=0.01,
                             help='Learning rate (default: 0.01)')
    train_parser.add_argument('--epsilon', type=float, default=0.3,
                             help='Initial exploration rate (default: 0.3)')
    train_parser.add_argument('--gamma', type=float, default=0.9,
                             help='Discount factor (default: 0.9)')
    train_parser.add_argument('--opponent', choices=['greedy'], 
                             default='greedy', help='Training opponent type')
    train_parser.add_argument('--eval-interval', type=int, default=50,
                             help='Evaluation interval (default: 50)')
    train_parser.add_argument('--save-interval', type=int, default=250,
                             help='Model save interval (default: 250)')
    train_parser.add_argument('--save-model', action='store_true',
                             help='Save trained model')
    train_parser.add_argument('--dictionary', default='dictionary.txt',
                             help='Dictionary file path')
    train_parser.add_argument('--multi-horizon', action='store_true',
                           help='Use multi-horizon learning')
    
    # Self-play training command
    if SelfPlayTrainer is not None:
        self_play_parser = subparsers.add_parser('self-play', help='Train agent using self-play')
        self_play_parser.add_argument('--episodes', type=int, default=2000,
                                     help='Number of self-play episodes (default: 2000)')
        self_play_parser.add_argument('--learning-rate', type=float, default=0.01,
                                     help='Learning rate (default: 0.01)')
        self_play_parser.add_argument('--epsilon', type=float, default=0.3,
                                     help='Initial exploration rate (default: 0.3)')
        self_play_parser.add_argument('--gamma', type=float, default=0.9,
                                     help='Discount factor (default: 0.9)')
        self_play_parser.add_argument('--buffer-size', type=int, default=5000,
                                     help='Experience replay buffer size (default: 5000)')
        self_play_parser.add_argument('--greedy-eval-interval', type=int, default=1,
                                     help='Evaluate vs greedy every N episodes (default: 1)')
        self_play_parser.add_argument('--greedy-eval-games', type=int, default=3,
                                     help='Games vs greedy per evaluation (default: 3)')
        self_play_parser.add_argument('--save-model', action='store_true',
                                     help='Save trained model')
        self_play_parser.add_argument('--dictionary', default='dictionary.txt',
                                     help='Dictionary file path')
        self_play_parser.add_argument('--multi-horizon', action='store_true',
                           help='Use multi-horizon learning')
    
    # human vs ai
    play_parser = subparsers.add_parser('play', help="Play against trained AI agent")
    play_parser.add_argument('--model-path', type=str, help="Path to AI model file")
    play_parser.add_argument('--dictionary', type=str, default='dictionary.txt', help="Path to dictionary file")

    args = parser.parse_args()

    if args.command == 'play':
        play_vs_human(args)
    else:
        parser.print_help()
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        print("🎮 Scrabble RL Agent")
        print("=" * 25)
        print("Available commands:")
        print("  train     - Train agent vs greedy opponent")
        if SelfPlayTrainer is not None:
            print("  self-play - Train agent using self-play")
        print("  play      - 🎮 Play against AI (GUI mode)")
        print("  evaluate  - Evaluate trained model")
        print("  analyze   - Analyze model behavior")
        print("  compare   - Compare multiple models")
        print("\nUse --help with any command for details")
        print("\n🎯 Quick start:")
        print("  python main.py train --episodes 500 --save-model")
        print("  python main.py play")
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_agent(args)
        elif args.command == 'self-play' and SelfPlayTrainer is not None:
            train_self_play_agent(args)
        elif args.command == 'play':
            play_vs_human(args)
        elif args.command == 'evaluate':
            evaluate_agent(args)
        elif args.command == 'analyze':
            analyze_agent(args)
        elif args.command == 'compare':
            compare_models(args)
        else:
            print(f"❌ Unknown or unavailable command: {args.command}")
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()