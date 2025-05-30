"""
Fixed Main entry point for Scrabble RL Agent
Updated imports to match actual class names
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# FIXED IMPORTS - use the actual class names from your files
from scrabble_agent import AdaptiveScrabbleQLearner, GreedyAgent, RandomAgent, HeuristicAgent
from trainer import EnhancedScrabbleTrainer  # Changed from ScrabbleTrainer
from evaluator import ScrabbleEvaluator
from utils import save_game_data

def train_agent(args):
    """Train a new RL agent"""
    print("Training Scrabble RL Agent")
    print("=" * 40)
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Opponent: {args.opponent}")
    print(f"Dictionary: {args.dictionary}")
    print()
    
    # Create agent with specified parameters
    agent = AdaptiveScrabbleQLearner(  # Using the actual class name
        num_features=8,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        gamma=args.gamma,
        buffer_size=2000,  # Add buffer parameters
        batch_size=32,
        target_update_frequency=100
    )
    
    # Create trainer - FIXED class name
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

def evaluate_agent(args):
    """Evaluate a trained agent"""
    print("Evaluating Scrabble RL Agent")
    print("=" * 40)
    print(f"Model: {args.model_path}")
    print(f"Games per opponent: {args.eval_games}")
    print(f"Dictionary: {args.dictionary}")
    print()
    
    # Load trained agent
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return None
    
    agent = AdaptiveScrabbleQLearner()  # Using actual class name
    agent.load_model(args.model_path)
    
    print(f"Loaded agent with {agent.training_episodes} training episodes")
    
    # Create evaluator
    evaluator = ScrabbleEvaluator(args.dictionary)
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(
        agent, 
        num_games_per_opponent=args.eval_games
    )
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed results
    results_path = f"evaluation_results_{timestamp}.json"
    save_game_data(results, results_path)
    print(f"\nDetailed results saved: {results_path}")
    
    # Analysis report
    analysis_path = f"evaluation_analysis_{timestamp}.json"
    evaluator.create_evaluation_report(results, analysis_path)
    
    # Print summary to console
    print_evaluation_summary(results)
    
    return results

def analyze_agent(args):
    """Deep analysis of agent behavior"""
    print("Analyzing Agent Strategic Behavior")
    print("=" * 40)
    
    # Load agent
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return None
    
    agent = AdaptiveScrabbleQLearner()  # Using actual class name
    agent.load_model(args.model_path)
    
    # Feature importance analysis
    print("\nFeature Importance Analysis:")
    print("-" * 30)
    feature_importance = agent.get_feature_importance()
    
    # Sort by absolute importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), reverse=True)
    
    for i, (feature_name, weight) in enumerate(sorted_features, 1):
        importance_bar = "█" * int(abs(weight) * 20) if abs(weight) > 0 else ""
        sign = "+" if weight >= 0 else "-"
        print(f"{i:2d}. {feature_name:20s} {sign}{abs(weight):6.3f} {importance_bar}")
    
    # Training statistics
    print(f"\nTraining Statistics:")
    print("-" * 20)
    stats = agent.get_training_stats()
    print(f"Training Episodes: {stats['training_episodes']}")
    print(f"Total Updates: {stats['total_updates']}")
    print(f"Current Epsilon: {stats['current_epsilon']:.3f}")
    
    # Strategic insights
    print(f"\nStrategic Insights:")
    print("-" * 18)
    
    # Analyze weight patterns
    weights = [w for w in feature_importance.values()]
    max_weight = max(abs(w) for w in weights) if weights else 0
    
    if max_weight > 0:
        dominant_features = [name for name, weight in feature_importance.items() 
                            if abs(weight) > max_weight * 0.7]
        
        if dominant_features:
            print(f"• Agent heavily prioritizes: {', '.join(dominant_features)}")
        
        negative_features = [name for name, weight in feature_importance.items() 
                            if weight < -0.1]
        
        if negative_features:
            print(f"• Agent learned to avoid: {', '.join(negative_features)}")
    
    return agent

def compare_agents(args):
    """Compare multiple trained agents"""
    print("Comparing Multiple Agents")
    print("=" * 30)
    
    if len(args.model_paths) < 2:
        print("Error: Need at least 2 model paths for comparison")
        return None
    
    # Load all agents
    agents = []
    for i, model_path in enumerate(args.model_paths):
        if not Path(model_path).exists():
            print(f"Warning: Model file not found: {model_path}")
            continue
        
        agent = AdaptiveScrabbleQLearner()  # Using actual class name
        agent.load_model(model_path)
        
        # Use filename as agent name
        agent_name = Path(model_path).stem
        agents.append((agent_name, agent))
        print(f"Loaded: {agent_name}")
    
    if len(agents) < 2:
        print("Error: Could not load enough agents for comparison")
        return None
    
    # Create evaluator and compare
    evaluator = ScrabbleEvaluator(args.dictionary)
    results = evaluator.compare_agents(agents, num_games=args.comparison_games)
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = f"agent_comparison_{timestamp}.json"
    save_game_data(results, comparison_path)
    print(f"\nComparison results saved: {comparison_path}")
    
    return results

def print_evaluation_summary(results):
    """Print formatted evaluation summary"""
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    # Overall performance
    summary = results.get('summary_stats', {})
    overall = summary.get('overall_performance', {})
    
    print(f"\nOverall Performance:")
    print(f"  Win Rate: {overall.get('win_rate', 0):>6.1%}")
    print(f"  Avg Score: {overall.get('avg_score', 0):>5.1f}")
    print(f"  Total Games: {overall.get('total_games', 0):>4d}")
    
    # Per-opponent breakdown
    print(f"\nPer-Opponent Results:")
    opponents = results.get('opponents', {})
    for opponent_name, opp_result in opponents.items():
        win_rate = opp_result.get('win_rate', 0)
        avg_score = opp_result.get('avg_agent_score', 0)
        score_gap = opp_result.get('avg_score_gap', 0)
        
        print(f"  {opponent_name:>10s}: {win_rate:5.1%} | "
              f"Score {avg_score:5.1f} | Gap {score_gap:+5.1f}")
    
    # Strategic insights
    strategic = results.get('strategic_analysis', {})
    insights = strategic.get('strategic_insights', [])
    
    if insights:
        print(f"\nStrategic Insights:")
        for insight in insights[:5]:  # Show top 5 insights
            print(f"  • {insight}")
    
    # Strengths and weaknesses
    strengths = summary.get('strengths', [])
    weaknesses = summary.get('weaknesses', [])
    
    if strengths:
        print(f"\nStrengths:")
        for strength in strengths:
            print(f"  ✓ {strength}")
    
    if weaknesses:
        print(f"\nAreas for Improvement:")
        for weakness in weaknesses:
            print(f"  ⚠ {weakness}")
    
    print("=" * 50)

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Scrabble RL Agent - Train, Evaluate, and Analyze')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new RL agent')
    train_parser.add_argument('--episodes', type=int, default=500,
                             help='Number of training episodes (default: 500)')
    train_parser.add_argument('--learning-rate', type=float, default=0.01,
                             help='Learning rate (default: 0.01)')
    train_parser.add_argument('--epsilon', type=float, default=0.3,
                             help='Initial exploration rate (default: 0.3)')
    train_parser.add_argument('--gamma', type=float, default=0.9,
                             help='Discount factor (default: 0.9)')
    train_parser.add_argument('--opponent', choices=['greedy', 'random', 'heuristic'], 
                             default='greedy', help='Training opponent type')
    train_parser.add_argument('--eval-interval', type=int, default=50,
                             help='Evaluation interval (default: 50)')
    train_parser.add_argument('--save-interval', type=int, default=250,
                             help='Model save interval (default: 250)')
    train_parser.add_argument('--save-model', action='store_true',
                             help='Save trained model')
    train_parser.add_argument('--dictionary', default='dictionary.txt',
                             help='Dictionary file path')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model-path', required=True,
                            help='Path to trained model file')
    eval_parser.add_argument('--eval-games', type=int, default=200,
                            help='Games per opponent (default: 200)')
    eval_parser.add_argument('--dictionary', default='dictionary.txt',
                            help='Dictionary file path')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze agent behavior')
    analyze_parser.add_argument('--model-path', required=True,
                               help='Path to trained model file')
    
    # Comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple agents')
    compare_parser.add_argument('--model-paths', nargs='+', required=True,
                               help='Paths to model files to compare')
    compare_parser.add_argument('--comparison-games', type=int, default=50,
                               help='Games per matchup (default: 50)')
    compare_parser.add_argument('--dictionary', default='dictionary.txt',
                               help='Dictionary file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_agent(args)
        elif args.command == 'evaluate':
            evaluate_agent(args)
        elif args.command == 'analyze':
            analyze_agent(args)
        elif args.command == 'compare':
            compare_agents(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()