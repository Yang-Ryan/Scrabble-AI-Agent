"""
Enhanced Main entry point for Scrabble RL Agent
Now supports both regular training and self-play training
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Import all necessary components
from scrabble_agent import AdaptiveScrabbleQLearner, GreedyAgent
from trainer import SelfPlayTrainer
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
        target_update_frequency=100
    )
    
    # Create regular trainer
    trainer = EnhancedScrabbleTrainer(args.dictionary)
    
    # Train agent
    trained_agent = trainer.train_agent(
        agent=agent,
        opponent_type=args.opponent,
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
        target_update_frequency=100
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

def analyze_agent(args):
    """Deep analysis of agent behavior"""
    print("🔍 AGENT ANALYSIS")
    print("=" * 40)
    
    # Load agent
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return None
    
    agent = AdaptiveScrabbleQLearner()
    agent.load_model(args.model_path)
    
    # Feature importance analysis
    print("\n📊 Feature Importance Analysis:")
    print("-" * 35)
    feature_importance = agent.get_feature_importance()
    
    # Sort by absolute importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), reverse=True)
    
    for i, (feature_name, weight) in enumerate(sorted_features, 1):
        importance_bar = "█" * int(abs(weight) * 20) if abs(weight) > 0 else ""
        sign = "+" if weight >= 0 else "-"
        print(f"{i:2d}. {feature_name:25s} {sign}{abs(weight):6.3f} {importance_bar}")
    
    # Training statistics
    print(f"\n📈 Training Statistics:")
    print("-" * 22)
    stats = agent.get_training_stats()
    print(f"Training Episodes: {stats['training_episodes']}")
    print(f"Total Updates: {stats['total_updates']}")
    print(f"Current Epsilon: {stats['current_epsilon']:.3f}")
    print(f"Buffer Size: {stats['buffer_size']}/{stats['buffer_max_size']}")
    
    # Adaptive learning stats
    print(f"\n🤖 Adaptive Learning:")
    print("-" * 19)
    timing_stats = agent.get_timing_stats()
    if timing_stats:
        urgency_levels = timing_stats.get('urgency_levels', [])
        transition_points = timing_stats.get('transition_points', [])
        print(f"Urgency Strategy: {[f'{u:.2f}' for u in urgency_levels]}")
        print(f"Transition Points: {[f'{t:.2f}' for t in transition_points]}")
    
    tile_stats = agent.get_tile_stats()
    if tile_stats:
        changed_tiles = tile_stats.get('significantly_changed_tiles', {})
        if changed_tiles:
            print(f"Learned Tile Values:")
            for tile, multiplier in sorted(changed_tiles.items(), 
                                         key=lambda x: abs(x[1] - 1.0), reverse=True)[:5]:
                print(f"  {tile}: {multiplier:.2f}x strategic value")
    
    return agent

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
    parser = argparse.ArgumentParser(description='Scrabble RL Agent - Enhanced Training & Analysis')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Regular training command
    train_parser = subparsers.add_parser('train', help='Train agent vs greedy opponent')
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
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model-path', required=True,
                            help='Path to trained model file')
    eval_parser.add_argument('--eval-games', type=int, default=100,
                            help='Number of evaluation games (default: 100)')
    eval_parser.add_argument('--dictionary', default='dictionary.txt',
                            help='Dictionary file path')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze agent behavior')
    analyze_parser.add_argument('--model-path', required=True,
                               help='Path to trained model file')
    
    # Comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--model-paths', nargs='+', required=True,
                               help='Paths to model files to compare')
    compare_parser.add_argument('--comparison-games', type=int, default=50,
                               help='Games per model vs greedy (default: 50)')
    compare_parser.add_argument('--dictionary', default='dictionary.txt',
                               help='Dictionary file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        print("🎮 Scrabble RL Agent")
        print("=" * 20)
        print("Available commands:")
        print("  train     - Train agent vs greedy opponent")
        print("  self-play - Train agent using self-play")
        print("  evaluate  - Evaluate trained model")
        print("  analyze   - Analyze model behavior")
        print("  compare   - Compare multiple models")
        print("\nUse --help with any command for details")
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_agent(args)
        elif args.command == 'self-play':
            train_self_play_agent(args)
        elif args.command == 'evaluate':
            evaluate_agent(args)
        elif args.command == 'analyze':
            analyze_agent(args)
        elif args.command == 'compare':
            compare_models(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()