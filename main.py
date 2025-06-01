import argparse
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

from scrabble_agent import AdaptiveScrabbleQLearner, GreedyAgent
from trainer import SelfPlayTrainer, ModelEvaluator
from utils import save_game_data

def train_agent(args):
    """Train agent against greedy opponent with consistent arguments"""
    print("🎯 GREEDY TRAINING MODE")
    print("=" * 40)
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Evaluation: Every {args.greedy_eval_interval} episodes")
    print(f"Games per Evaluation: {args.greedy_eval_games}")
    print(f"Dictionary: {args.dictionary}")
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
    
    # Create trainer
    trainer = SelfPlayTrainer(args.dictionary)
    
    # Train agent against greedy opponent
    trained_agent = trainer._train_vs_greedy(
        agent=agent,
        num_episodes=args.episodes,
        greedy_eval_interval=args.greedy_eval_interval,
        greedy_eval_games=args.greedy_eval_games,
        verbose=True
    )
    
    # Save trained model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"greedy_model_{timestamp}_ep{args.episodes}.json"
        trained_agent.save_model(model_path)
        print(f"\nTrained model saved: {model_path}")
        
        # Save training data
        training_data_path = f"greedy_training_data_{timestamp}.json"
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

def play_vs_human(args):
    """Launch Human vs AI game interface"""
    print("🎮 LAUNCHING HUMAN VS AI GAME")
    print("=" * 40)

    try:
        from scrabble_gui import ScrabbleGameGUI  # Import here to avoid dependency issues
        
        game = ScrabbleGameGUI(dictionary_path=args.dictionary)

        # Auto-load model if specified
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

    except ImportError:
        print("❌ GUI module not available. Please ensure scrabble_gui.py is present.")
    except Exception as e:
        print(f"❌ Error launching game: {e}")
        import traceback
        traceback.print_exc()

def evaluate_agent(args):
    """Evaluate a trained agent against greedy opponent"""
    print("📊 AGENT EVALUATION")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Games: {args.games}")
    print(f"Dictionary: {args.dictionary}")
    if args.plot:
        print("Plotting: Enabled")
    elif args.no_plot:
        print("Plotting: Disabled")
    else:
        print("Plotting: Auto (enabled for 50+ games)")
    print("=" * 50)
    
    # Load trained agent
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model file not found: {args.model_path}")
        return None
    
    try:
        agent = AdaptiveScrabbleQLearner()
        agent.load_model(args.model_path)
        print(f"✅ Loaded agent with {agent.training_episodes} training episodes")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create evaluator
    evaluator = ModelEvaluator(args.dictionary)
    
    # Run evaluation
    print(f"\n🎮 Starting evaluation: {args.games} games vs Greedy...")
    results = evaluator.evaluate_vs_greedy(
        agent=agent, 
        num_games=args.games,
        verbose=args.verbose
    )
    
    # Print results
    print_evaluation_summary(results)
    
    # Generate plots
    should_plot = False
    if args.plot:
        should_plot = True
    elif not args.no_plot and args.games >= 50:
        should_plot = True
        print(f"\n📊 Generating evaluation plots (use --no-plot to skip)...")
    
    if should_plot:
        try:
            plot_path = evaluator.plot_evaluation_results(results, save_plot=True)
            if plot_path:
                print(f"📈 Comprehensive evaluation plots saved: {plot_path}")
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
    
    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"evaluation_results_{timestamp}.json"
        save_game_data(results, results_path)
        print(f"\n💾 Results saved: {results_path}")
    
    return results

def print_evaluation_summary(results):
    """Print formatted evaluation summary"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Performance Metrics:")
    print(f"  Total Games Played: {results.get('games_played', 0):>6d}")
    print(f"  Wins: {results.get('wins', 0):>6d}")
    print(f"  Losses: {results.get('games_played', 0) - results.get('wins', 0):>6d}")
    print(f"  Win Rate: {results.get('win_rate', 0):>6.1%}")
    
    print(f"\n🎯 Score Analysis:")
    print(f"  Agent Average Score: {results.get('avg_score', 0):>6.1f}")
    print(f"  Greedy Average Score: {results.get('avg_opponent_score', 0):>6.1f}")
    print(f"  Score Gap (Agent - Greedy): {results.get('avg_score_gap', 0):>+6.1f}")
    
    # Performance assessment
    win_rate = results.get('win_rate', 0)
    print(f"\n🏆 Performance Assessment:")
    if win_rate > 0.75:
        print("  🚀 EXCELLENT - Dominates greedy opponent!")
    elif win_rate > 0.65:
        print("  ✅ VERY GOOD - Consistently beats greedy opponent")
    elif win_rate > 0.55:
        print("  👍 GOOD - Usually beats greedy opponent")
    elif win_rate > 0.45:
        print("  📊 COMPETITIVE - Evenly matched with greedy")
    elif win_rate > 0.35:
        print("  ⚠️ BELOW AVERAGE - Struggles against greedy")
    else:
        print("  ❌ POOR - Loses consistently to greedy")
    
    # Score analysis
    score_gap = results.get('avg_score_gap', 0)
    if abs(score_gap) < 5:
        print("  🔍 Score margins are very close")
    elif score_gap > 15:
        print("  💪 Winning by significant score margins")
    elif score_gap > 5:
        print("  📈 Winning by comfortable margins")
    elif score_gap < -15:
        print("  📉 Losing by significant margins")
    elif score_gap < -5:
        print("  ⚠️ Losing by noticeable margins")
    
    # Additional insights
    if 'detailed_results' in results:
        games = results['detailed_results']
        
        # Biggest win/loss
        wins_only = [g['score_gap'] for g in games if g['agent_won']]
        losses_only = [g['score_gap'] for g in games if not g['agent_won']]
        
        biggest_win = max(wins_only) if wins_only else 0
        biggest_loss = min(losses_only) if losses_only else 0
        
        print(f"\n🎲 Game Details:")
        if biggest_win > 0:
            print(f"  Biggest Victory Margin: +{biggest_win}")
        if biggest_loss < 0:
            print(f"  Biggest Defeat Margin: {biggest_loss}")
        
        # Score distribution
        agent_scores = [g['agent_score'] for g in games]
        if agent_scores:
            print(f"  Agent Score Range: {min(agent_scores)} - {max(agent_scores)}")
            print(f"  Agent Score Std Dev: {np.std(agent_scores):.1f}")
    
    print("="*60)


def main():
    """Main entry point with enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='🎮 Scrabble RL Agent - Training, Evaluation & Human Play')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Regular training command (updated to match self-play pattern)
    train_parser = subparsers.add_parser('train', help='Train agent vs greedy opponent')
    train_parser.add_argument('--episodes', type=int, default=2000,
                             help='Number of training episodes (default: 2000)')
    train_parser.add_argument('--learning-rate', type=float, default=0.01,
                             help='Learning rate (default: 0.01)')
    train_parser.add_argument('--epsilon', type=float, default=0.3,
                             help='Initial exploration rate (default: 0.3)')
    train_parser.add_argument('--gamma', type=float, default=0.9,
                             help='Discount factor (default: 0.9)')
    train_parser.add_argument('--buffer-size', type=int, default=5000,
                             help='Experience replay buffer size (default: 5000)')
    train_parser.add_argument('--greedy-eval-interval', type=int, default=1,
                             help='Evaluate vs greedy every N episodes (default: 1)')
    train_parser.add_argument('--greedy-eval-games', type=int, default=3,
                             help='Games vs greedy per evaluation (default: 3)')
    train_parser.add_argument('--save-model', action='store_true',
                             help='Save trained model')
    train_parser.add_argument('--dictionary', default='dictionary.txt',
                             help='Dictionary file path')
    train_parser.add_argument('--multi-horizon', action='store_true',
                             help='Use multi-horizon learning')

    # Self-play training command
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

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model against greedy opponent')
    eval_parser.add_argument('games', type=int, 
                            help='Number of games to play (e.g., 300)')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model file')
    eval_parser.add_argument('--dictionary', default='dictionary.txt',
                            help='Dictionary file path')
    eval_parser.add_argument('--verbose', action='store_true',
                            help='Show detailed game-by-game results')
    eval_parser.add_argument('--save-results', action='store_true',
                            help='Save evaluation results to file')
    eval_parser.add_argument('--plot', action='store_true',
                            help='Generate comprehensive evaluation plots')
    eval_parser.add_argument('--no-plot', action='store_true',
                            help='Skip plot generation (faster evaluation)')
    
    # Human vs AI
    play_parser = subparsers.add_parser('play', help="Play against trained AI agent")
    play_parser.add_argument('--model-path', type=str, help="Path to AI model file")
    play_parser.add_argument('--dictionary', type=str, default='dictionary.txt', help="Path to dictionary file")

    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        print("🎮 Scrabble RL Agent")
        print("=" * 25)
        print("Available commands:")
        print("  train     - Train agent vs greedy opponent")
        print("  self-play - Train agent using self-play")
        print("  evaluate  - 📊 Evaluate trained model performance")
        print("  play      - 🎮 Play against AI (GUI mode)")
        print("\nUse --help with any command for details")
        print("\n🎯 Quick start:")
        print("  python main.py train --episodes 500 --save-model")
        print("  python main.py evaluate 300 --model-path your_model.json")
        print("  python main.py play")
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_agent(args)
        elif args.command == 'self-play':
            train_self_play_agent(args)
        elif args.command == 'evaluate':
            evaluate_agent(args)
        elif args.command == 'play':
            play_vs_human(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()