from tqdm import tqdm
import argparse
import json
import os
import time
from datetime import datetime

from scrabble_game import ScrabbleGame
from rl_agent import QLearningAgent
from baseline_agent import RandomAgent, GreedyAgent, HeuristicAgent, AdaptiveAgent
from evaluation import GameEvaluator

def create_word_dictionary():
    """Create a basic word dictionary for testing"""
    # Extended word list for testing
    words = [
        # Common short words
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
        'OUR', 'HAD', 'BUT', 'WHAT', 'SO', 'UP', 'OUT', 'IF', 'ABOUT', 'WHO', 'GET', 'WHICH',
        'GO', 'ME', 'WHEN', 'MAKE', 'CAN', 'LIKE', 'TIME', 'NO', 'JUST', 'HIM', 'KNOW', 'TAKE',
        'PEOPLE', 'INTO', 'YEAR', 'YOUR', 'GOOD', 'SOME', 'COULD', 'THEM', 'SEE', 'OTHER',
        
        # Gaming and AI words
        'GAME', 'PLAY', 'WIN', 'LOSE', 'SCORE', 'POINT', 'TURN', 'MOVE', 'RULE', 'BOARD',
        'TILE', 'WORD', 'LETTER', 'AGENT', 'LEARN', 'SMART', 'THINK', 'PLAN', 'STRATEGY',
        
        # Common Scrabble words
        'QI', 'XI', 'XU', 'ZA', 'ZO', 'JO', 'KI', 'OX', 'EX', 'AX', 'MY', 'BY', 'OF', 'TO',
        'IN', 'IT', 'ON', 'AS', 'AT', 'BE', 'OR', 'AN', 'IS', 'WE', 'DO', 'HE', 'US', 'AM',
        
        # Medium length words
        'QUICK', 'BROWN', 'JUMPS', 'OVER', 'LAZY', 'FOXES', 'ZEBRA', 'QUEEN', 'QUILT',
        'JAZZ', 'FIZZ', 'BUZZ', 'MAZE', 'HAZE', 'BLAZE', 'PRIZE', 'FROZE', 'GAZE',
        'APPLE', 'GRAPE', 'ORANGE', 'MANGO', 'PEACH', 'BERRY', 'MELON', 'LEMON',
        'HOUSE', 'MOUSE', 'HORSE', 'GOOSE', 'MOOSE', 'LOOSE', 'NOOSE', 'GOOSE',
        'WATER', 'FIRE', 'EARTH', 'WIND', 'STORM', 'CLOUD', 'SUNNY', 'RAINY',
        
        # Longer words for variety
        'PYTHON', 'COMPUTER', 'SCIENCE', 'MACHINE', 'LEARNING', 'NEURAL', 'NETWORK',
        'ALGORITHM', 'FUNCTION', 'VARIABLE', 'BOOLEAN', 'INTEGER', 'STRING', 'ARRAY',
        'OBJECT', 'METHOD', 'CLASS', 'MODULE', 'IMPORT', 'RETURN', 'PRINT', 'INPUT',
        'TRAINING', 'TESTING', 'VALIDATION', 'ACCURACY', 'PRECISION', 'RECALL',
        'RESEARCH', 'PROJECT', 'EXPERIMENT', 'ANALYSIS', 'RESULTS', 'CONCLUSION'
    ]
    
    # Write to file
    with open('dictionary.txt', 'w') as f:
        for word in sorted(set(words)):
            f.write(word.upper() + '\n')
    
    print(f"Created dictionary with {len(set(words))} words")
    return 'dictionary.txt'

def train_agent(args):
    """Train the RL agent with visual progress bar"""
    print(f"Training {args.episodes} episodes vs {args.training_opponent} agent...")

    # Create dictionary if needed
    if not os.path.exists(args.dictionary):
        print("Dictionary not found, creating basic dictionary...")
        args.dictionary = create_word_dictionary()

    # Initialize RL agent
    rl_agent = QLearningAgent(
        name="RL_Agent",
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay
    )

    # Load existing model if specified
    if args.load_model:
        rl_agent.load_model(args.load_model)

    # Initialize training opponent
    opponents = {
        'greedy': GreedyAgent("Training_Opponent"),
        'heuristic': HeuristicAgent("Training_Opponent"),
        'adaptive': AdaptiveAgent("Training_Opponent"),
        'random': RandomAgent("Training_Opponent")
    }
    opponent = opponents.get(args.training_opponent, RandomAgent("Training_Opponent"))

    # Training loop
    start_time = time.time()
    training_stats = {
        'episodes': [],
        'scores': [],
        'wins': [],
        'epsilon_values': [],
        'avg_move_scores': []
    }

    wins = 0
    total_score = 0

    pbar = tqdm(range(args.episodes), desc="Training Progress")

    for episode in pbar:
        # Create new game
        game = ScrabbleGame(args.dictionary)

        # Alternate starting player
        if episode % 2 == 0:
            game.add_player(rl_agent)
            game.add_player(opponent)
        else:
            game.add_player(opponent)
            game.add_player(rl_agent)

        rl_agent.initialize_opponent_model(game.tile_bag.letter_distribution)

        # Play game and get result
        game_result = play_training_game(game, rl_agent, opponent)

        # Update stats
        total_score += rl_agent.score
        if game_result['winner'] == rl_agent.name:
            wins += 1

        if episode % 10 == 0:
            training_stats['episodes'].append(episode)
            training_stats['scores'].append(rl_agent.score)
            training_stats['wins'].append(wins / (episode + 1))
            training_stats['epsilon_values'].append(rl_agent.epsilon)
            training_stats['avg_move_scores'].append(game_result.get('avg_move_score', 0))

        if episode % 100 == 0 and episode > 0:
            win_rate = wins / episode
            avg_score = total_score / episode
            pbar.set_postfix({
                'Win Rate': f'{win_rate:.1%}',
                'Avg Score': f'{avg_score:.1f}',
                'Epsilon': f'{rl_agent.epsilon:.3f}',
                'Q-Table': len(rl_agent.q_table)
            })

    # Final summary
    final_win_rate = wins / args.episodes
    final_avg_score = total_score / args.episodes
    training_time = time.time() - start_time

    print("\nTraining Summary:")
    print(f"  Episodes       : {args.episodes}")
    print(f"  Final Win Rate : {final_win_rate:.1%}")
    print(f"  Final Avg Score: {final_avg_score:.1f}")
    print(f"  Final Epsilon  : {rl_agent.epsilon:.3f}")
    print(f"  Q-table Size   : {len(rl_agent.q_table)}")
    print(f"  Time Elapsed   : {training_time / 60:.1f} min")

    # Save model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"rl_model_{timestamp}.json"
        rl_agent.save_model(model_path)
        print(f"Model saved to: {model_path}")

    # Save training stats
    stats_path = f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"Training statistics saved to: {stats_path}")

    return rl_agent


def play_training_game(game: ScrabbleGame, rl_agent: QLearningAgent, opponent) -> dict:
    """Play a single training game and return results"""
    turn_count = 0
    max_turns = 200
    move_scores = []
    
    while not game.game_over and turn_count < max_turns:
        current_player = game.players[game.current_player]
        
        # Store previous state for RL agent
        if current_player == rl_agent:
            prev_state = rl_agent.get_state_representation(game)
        
        # Get and execute move
        move = current_player.get_move(game)
        
        if move:
            success = game.make_move(game.current_player, move)
            if success and current_player == rl_agent:
                move_scores.append(move['score'])
                
                # Calculate reward and update Q-value
                reward = rl_agent.calculate_reward(game, move, {})
                next_state = rl_agent.get_state_representation(game)
                rl_agent.update_q_value(reward, next_state)
        
        # Switch players
        game.current_player = (game.current_player + 1) % len(game.players)
        turn_count += 1
        
        # Check end condition
        if game.tile_bag.remaining_count() == 0:
            if any(len(player.tiles) == 0 for player in game.players):
                game.game_over = True
    
    # Determine winner and final update
    winner = max(game.players, key=lambda p: p.score)
    won = winner == rl_agent
    final_reward = rl_agent.score - opponent.score
    
    rl_agent.end_game_update(final_reward, won)
    
    return {
        'winner': winner.name,
        'rl_score': rl_agent.score,
        'opponent_score': opponent.score,
        'turns': turn_count,
        'avg_move_score': sum(move_scores) / len(move_scores) if move_scores else 0
    }

def evaluate_agent(args):
    """Evaluate the trained RL agent"""
    print("=" * 60)
    print("SCRABBLE RL AGENT EVALUATION")
    print("Team 34: Young Meh Meh")
    print("=" * 60)
    
    # Load trained agent
    if not args.model_path:
        print("Error: Model path required for evaluation")
        return
    
    rl_agent = QLearningAgent("RL_Agent_Eval")
    rl_agent.load_model(args.model_path)
    
    # Set epsilon to 0 for evaluation (no exploration)
    rl_agent.epsilon = 0.0
    
    # Create dictionary if needed
    if not os.path.exists(args.dictionary):
        print("Dictionary not found, creating basic dictionary...")
        args.dictionary = create_word_dictionary()
    
    # Initialize evaluator
    evaluator = GameEvaluator(args.dictionary)
    
    print(f"Evaluation Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Games per baseline: {args.eval_games}")
    print(f"  Dictionary: {args.dictionary}")
    print()
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_agent_vs_baselines(
        rl_agent, 
        num_games=args.eval_games,
        save_results=True
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    analysis = evaluation_results['analysis']
    overall_perf = analysis['overall_performance']
    
    print(f"Overall Performance:")
    print(f"  Total Games: {overall_perf['total_games']}")
    print(f"  Total Wins: {overall_perf['total_wins']}")
    print(f"  Overall Win Rate: {overall_perf['overall_win_rate']:.1%}")
    print(f"  Average Score: {overall_perf['avg_score_per_game']:.1f}")
    print()
    
    print("Performance vs Each Baseline:")
    for baseline, perf in analysis['performance_trends'].items():
        print(f"  {baseline}:")
        print(f"    Win Rate: {perf['win_rate']:.1%}")
        print(f"    Avg Score Diff: {perf['avg_score_difference']:.1f}")
        print(f"    Performance Level: {perf['performance_level']}")
    print()
    
    if analysis['strengths']:
        print("Strengths:")
        for strength in analysis['strengths']:
            print(f"  • {strength}")
    
    if analysis['weaknesses']:
        print("\nWeaknesses:")
        for weakness in analysis['weaknesses']:
            print(f"  • {weakness}")
    
    return evaluation_results

def ablation_study(args):
    """Run ablation study on RL agent components"""
    print("=" * 60)
    print("ABLATION STUDY")
    print("Team 34: Young Meh Meh")
    print("=" * 60)
    
    if not args.model_path:
        print("Error: Model path required for ablation study")
        return
    
    # Load base agent
    base_agent = QLearningAgent("Base_Agent")
    base_agent.load_model(args.model_path)
    
    # Create dictionary if needed
    if not os.path.exists(args.dictionary):
        args.dictionary = create_word_dictionary()
    
    # Define modifications to test
    modifications = {
        'no_opponent_modeling': {
            'opponent_model': None
        },
        'high_exploration': {
            'epsilon': 0.3
        },
        'low_learning_rate': {
            'learning_rate': 0.01
        },
        'no_defensive_weight': {
            'defensive_weight': 0.0,
            'offensive_weight': 1.0
        },
        'pure_greedy': {
            'epsilon': 0.0,
            'exploration_bonus': 0.0
        }
    }
    
    # Initialize evaluator
    evaluator = GameEvaluator(args.dictionary)
    
    # Run ablation study
    results = evaluator.ablation_study(
        base_agent, 
        modifications, 
        num_games=args.ablation_games
    )
    
    # Print results
    print("Ablation Study Results:")
    print("-" * 40)
    
    # Sort by win rate
    sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    
    for mod_name, result in sorted_results:
        print(f"{mod_name}:")
        print(f"  Win Rate: {result['win_rate']:.1%}")
        print(f"  Avg Score Diff: {result['avg_score_diff']:.1f}")
        print(f"  Parameters: {result['parameters']}")
        print()
    
    return results

def demo_game(args):
    """Run a demonstration game"""
    print("=" * 60)
    print("DEMONSTRATION GAME")
    print("Team 34: Young Meh Meh")
    print("=" * 60)
    
    # Create dictionary if needed
    if not os.path.exists(args.dictionary):
        args.dictionary = create_word_dictionary()
    
    # Initialize agents
    if args.model_path:
        rl_agent = QLearningAgent("RL_Agent")
        rl_agent.load_model(args.model_path)
        rl_agent.epsilon = 0.0  # No exploration for demo
    else:
        rl_agent = QLearningAgent("RL_Agent")
    
    opponent = HeuristicAgent("Heuristic_Opponent")
    
    # Create and setup game
    game = ScrabbleGame(args.dictionary)
    game.add_player(rl_agent)
    game.add_player(opponent)
    
    rl_agent.initialize_opponent_model(game.tile_bag.letter_distribution)
    
    print(f"Demo Game: {rl_agent.name} vs {opponent.name}")
    print("=" * 40)
    
    # Play game with detailed output
    turn_count = 0
    max_turns = 100
    
    while not game.game_over and turn_count < max_turns:
        current_player = game.players[game.current_player]
        
        print(f"\nTurn {turn_count + 1}: {current_player.name}")
        print(f"Current Scores - {rl_agent.name}: {rl_agent.score}, {opponent.name}: {opponent.score}")
        print(f"Tiles: {current_player.tiles}")
        
        # Get move
        move = current_player.get_move(game)
        
        if move:
            print(f"Plays: {move['word']} at {move['position']} ({move['direction']}) for {move['score']} points")
            success = game.make_move(game.current_player, move)
            if not success:
                print("Invalid move!")
        else:
            print("Passes turn")
        
        # Show board state (simplified)
        if turn_count % 5 == 0:
            print("\nCurrent board state (occupied squares):")
            occupied = []
            for i in range(len(game.board.board)):
                for j in range(len(game.board.board[i])):
                    if game.board.board[i][j] != '':
                        occupied.append(f"({i},{j}):{game.board.board[i][j]}")
            print(f"Occupied: {', '.join(occupied[:10])}{'...' if len(occupied) > 10 else ''}")
        
        # Switch players
        game.current_player = (game.current_player + 1) % len(game.players)
        turn_count += 1
        
        # Check end condition
        if game.tile_bag.remaining_count() == 0:
            if any(len(player.tiles) == 0 for player in game.players):
                game.game_over = True
    
    # Final results
    print("\n" + "=" * 40)
    print("GAME COMPLETED")
    print("=" * 40)
    print(f"Final Scores:")
    print(f"  {rl_agent.name}: {rl_agent.score}")
    print(f"  {opponent.name}: {opponent.score}")
    
    winner = max(game.players, key=lambda p: p.score)
    print(f"Winner: {winner.name}")
    print(f"Game Length: {turn_count} turns")
    print(f"Tiles Remaining: {game.tile_bag.remaining_count()}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Scrabble RL Agent - Team 34: Young Meh Meh')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    train_parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=0.3, help='Initial epsilon for exploration')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    train_parser.add_argument('--training-opponent', choices=['random', 'greedy', 'heuristic', 'adaptive'], 
                             default='greedy', help='Training opponent type')
    train_parser.add_argument('--dictionary', default='dictionary.txt', help='Dictionary file path')
    train_parser.add_argument('--save-model', action='store_true', help='Save trained model')
    train_parser.add_argument('--load-model', help='Load existing model to continue training')
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the trained agent')
    eval_parser.add_argument('--model-path', required=True, help='Path to trained model')
    eval_parser.add_argument('--eval-games', type=int, default=100, help='Games per baseline for evaluation')
    eval_parser.add_argument('--dictionary', default='dictionary.txt', help='Dictionary file path')
    
    # Ablation study arguments
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('--model-path', required=True, help='Path to trained model')
    ablation_parser.add_argument('--ablation-games', type=int, default=50, help='Games per modification')
    ablation_parser.add_argument('--dictionary', default='dictionary.txt', help='Dictionary file path')
    
    # Demo arguments
    demo_parser = subparsers.add_parser('demo', help='Run demonstration game')
    demo_parser.add_argument('--model-path', help='Path to trained model (optional)')
    demo_parser.add_argument('--dictionary', default='dictionary.txt', help='Dictionary file path')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_agent(args)
    elif args.command == 'evaluate':
        evaluate_agent(args)
    elif args.command == 'ablation':
        ablation_study(args)
    elif args.command == 'demo':
        demo_game(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()