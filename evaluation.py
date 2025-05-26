import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import json
from datetime import datetime
from collections import defaultdict
import seaborn as sns

from scrabble_game import ScrabbleGame
from rl_agent import QLearningAgent
from baseline_agent import RandomAgent, GreedyAgent, HeuristicAgent, MinimaxAgent, AdaptiveAgent


def convert_json_safe(obj):
    if isinstance(obj, (np.int64, np.int32, np.int_, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


class GameEvaluator:
    """Comprehensive evaluation system for Scrabble agents"""
    
    def __init__(self, dictionary_file: Optional[str] = None):
        self.dictionary_file = dictionary_file
        self.results = defaultdict(list)
        self.game_logs = []
        
    def evaluate_agent_vs_baselines(self, rl_agent: QLearningAgent, 
                                   num_games: int = 100, 
                                   save_results: bool = True) -> Dict:
        """Evaluate RL agent against multiple baseline agents"""
        
        baseline_agents = {
            'Random': RandomAgent,
            'Greedy': GreedyAgent,
            'Heuristic': HeuristicAgent,
            'Adaptive': AdaptiveAgent
        }
        
        print("Starting comprehensive evaluation...")
        print(f"RL Agent: {rl_agent.name}")
        print(f"Games per baseline: {num_games}")
        print("=" * 50)
        
        all_results = {}
        
        for baseline_name, baseline_class in baseline_agents.items():
            print(f"\nEvaluating against {baseline_name} Agent...")
            
            results = self._evaluate_head_to_head(
                rl_agent, 
                baseline_class(f"{baseline_name}_Baseline"),
                num_games
            )
            
            all_results[baseline_name] = results
            
            # Print intermediate results
            win_rate = results['rl_wins'] / num_games * 100
            avg_score_diff = np.mean(results['score_differences'])
            
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Avg Score Difference: {avg_score_diff:.1f}")
            print(f"  Avg Game Length: {np.mean(results['game_lengths']):.1f} turns")
        
        # Generate comprehensive analysis
        analysis = self._analyze_results(all_results)
        
        if save_results:
            self._save_evaluation_results(all_results, analysis)
        
        return {
            'detailed_results': all_results,
            'analysis': analysis
        }
    
    def _evaluate_head_to_head(self, agent1, agent2, num_games: int) -> Dict:
        """Evaluate two agents head-to-head"""
        results = {
            'rl_wins': 0,
            'baseline_wins': 0,
            'ties': 0,
            'rl_scores': [],
            'baseline_scores': [],
            'score_differences': [],
            'game_lengths': [],
            'rl_avg_move_time': [],
            'baseline_avg_move_time': [],
            'games_data': []
        }
        
        for game_num in range(num_games):
            if game_num % 10 == 0:
                print(f"  Game {game_num + 1}/{num_games}")
            
            # Create new game
            game = ScrabbleGame(self.dictionary_file)
            
            # Alternate who goes first
            if game_num % 2 == 0:
                game.add_player(agent1)
                game.add_player(agent2)
            else:
                game.add_player(agent2)
                game.add_player(agent1)
            
            # Initialize opponent modeling for RL agent
            if hasattr(agent1, 'initialize_opponent_model'):
                agent1.initialize_opponent_model(game.tile_bag.letter_distribution)
            if hasattr(agent2, 'initialize_opponent_model'):
                agent2.initialize_opponent_model(game.tile_bag.letter_distribution)
            
            # Play game and collect metrics
            game_data = self._play_game_with_metrics(game, agent1, agent2)
            
            # Record results
            if game_data['winner'] == agent1.name:
                results['rl_wins'] += 1
            elif game_data['winner'] == agent2.name:
                results['baseline_wins'] += 1
            else:
                results['ties'] += 1
            
            results['rl_scores'].append(game_data['rl_score'])
            results['baseline_scores'].append(game_data['baseline_score'])
            results['score_differences'].append(game_data['rl_score'] - game_data['baseline_score'])
            results['game_lengths'].append(game_data['turns'])
            results['rl_avg_move_time'].append(game_data['rl_avg_move_time'])
            results['baseline_avg_move_time'].append(game_data['baseline_avg_move_time'])
            results['games_data'].append(game_data)
            
            # Update RL agent with game outcome
            if hasattr(agent1, 'end_game_update'):
                won = game_data['winner'] == agent1.name
                final_reward = game_data['rl_score'] - game_data['baseline_score']
                agent1.end_game_update(final_reward, won)
        
        return results
    
    def _play_game_with_metrics(self, game: ScrabbleGame, agent1, agent2) -> Dict:
        """Play a game and collect detailed metrics"""
        start_time = time.time()
        turn_count = 0
        max_turns = 200  # Prevent infinite games
        
        move_times = {agent1.name: [], agent2.name: []}
        move_scores = {agent1.name: [], agent2.name: []}
        
        rl_agent = agent1 if isinstance(agent1, QLearningAgent) else agent2
        baseline_agent = agent2 if isinstance(agent1, QLearningAgent) else agent1
        
        board_states = []
        
        while not game.game_over and turn_count < max_turns:
            current_player = game.players[game.current_player]
            
            # Record board state
            board_states.append([row[:] for row in game.board.board])
            
            # Get move with timing
            move_start = time.time()
            move = current_player.get_move(game)
            move_time = time.time() - move_start
            
            move_times[current_player.name].append(move_time)
            
            if move:
                # Execute move
                success = game.make_move(game.current_player, move)
                if success:
                    move_scores[current_player.name].append(move['score'])
                    
                    # Update opponent model if applicable
                    if hasattr(rl_agent, 'update_opponent_model') and current_player != rl_agent:
                        board_before = board_states[-1] if board_states else None
                        board_after = [row[:] for row in game.board.board]
                        rl_agent.update_opponent_model(move, board_before, board_after)
                else:
                    move_scores[current_player.name].append(0)
            else:
                move_scores[current_player.name].append(0)
            
            # Switch to next player
            game.current_player = (game.current_player + 1) % len(game.players)
            turn_count += 1
            
            # Check end conditions
            if game.tile_bag.remaining_count() == 0:
                # Check if any player has no tiles
                if any(len(player.tiles) == 0 for player in game.players):
                    game.game_over = True
        
        # Determine winner
        winner = max(game.players, key=lambda p: p.score)
        
        # Collect game data
        game_data = {
            'winner': winner.name if winner.score > 0 else 'tie',
            'rl_score': rl_agent.score,
            'baseline_score': baseline_agent.score,
            'turns': turn_count,
            'duration': time.time() - start_time,
            'rl_avg_move_time': np.mean(move_times[rl_agent.name]) if move_times[rl_agent.name] else 0,
            'baseline_avg_move_time': np.mean(move_times[baseline_agent.name]) if move_times[baseline_agent.name] else 0,
            'rl_avg_move_score': np.mean(move_scores[rl_agent.name]) if move_scores[rl_agent.name] else 0,
            'baseline_avg_move_score': np.mean(move_scores[baseline_agent.name]) if move_scores[baseline_agent.name] else 0,
            'total_tiles_used': sum(len(move_scores[player.name]) for player in game.players),
            'board_utilization': self._calculate_board_utilization(game.board.board)
        }
        
        return game_data
    
    def _calculate_board_utilization(self, board: List[List[str]]) -> float:
        """Calculate percentage of board squares used"""
        total_squares = len(board) * len(board[0])
        used_squares = sum(1 for row in board for cell in row if cell != '')
        return used_squares / total_squares
    
    def _analyze_results(self, all_results: Dict) -> Dict:
        """Analyze evaluation results and generate insights"""
        analysis = {
            'overall_performance': {},
            'strengths': [],
            'weaknesses': [],
            'improvement_areas': [],
            'statistical_significance': {},
            'performance_trends': {}
        }
        
        # Overall performance metrics
        total_games = sum(len(results['rl_scores']) for results in all_results.values())
        total_wins = sum(results['rl_wins'] for results in all_results.values())
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        
        analysis['overall_performance'] = {
            'total_games': total_games,
            'total_wins': total_wins,
            'overall_win_rate': overall_win_rate,
            'avg_score_per_game': np.mean([
                score for results in all_results.values() 
                for score in results['rl_scores']
            ])
        }
        
        # Performance against each baseline
        for baseline_name, results in all_results.items():
            win_rate = results['rl_wins'] / len(results['rl_scores'])
            avg_score_diff = np.mean(results['score_differences'])
            
            performance_level = 'excellent' if win_rate > 0.8 else (
                'good' if win_rate > 0.6 else (
                    'fair' if win_rate > 0.4 else 'poor'
                )
            )
            
            analysis['performance_trends'][baseline_name] = {
                'win_rate': win_rate,
                'avg_score_difference': avg_score_diff,
                'performance_level': performance_level,
                'consistency': np.std(results['score_differences'])
            }
        
        # Identify strengths and weaknesses
        best_performance = max(analysis['performance_trends'].items(), 
                             key=lambda x: x[1]['win_rate'])
        worst_performance = min(analysis['performance_trends'].items(), 
                              key=lambda x: x[1]['win_rate'])
        
        analysis['strengths'].append(f"Strong performance against {best_performance[0]} "
                                   f"(win rate: {best_performance[1]['win_rate']:.1%})")
        
        if worst_performance[1]['win_rate'] < 0.5:
            analysis['weaknesses'].append(f"Struggles against {worst_performance[0]} "
                                        f"(win rate: {worst_performance[1]['win_rate']:.1%})")
        
        # Statistical significance testing
        for baseline_name, results in all_results.items():
            # Simple t-test for score differences
            score_diffs = np.array(results['score_differences'])
            t_stat = np.mean(score_diffs) / (np.std(score_diffs) / np.sqrt(len(score_diffs)))
            
            analysis['statistical_significance'][baseline_name] = {
                't_statistic': t_stat,
                'significant': abs(t_stat) > 1.96,  # 95% confidence
                'mean_difference': np.mean(score_diffs),
                'std_error': np.std(score_diffs) / np.sqrt(len(score_diffs))
            }
        
        return analysis

    def _save_evaluation_results(self, results: Dict, analysis: Dict):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for baseline, data in results.items():
                json_results[baseline] = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in data.items()
                    if key != 'games_data'  # Skip detailed game data
                }
            json.dump(analysis, f, indent=2, default=convert_json_safe)
        
        # Save analysis
        analysis_file = f"evaluation_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=convert_json_safe)
        
        # Generate and save plots
        self._generate_evaluation_plots(results, analysis, timestamp)
        
        print(f"\nResults saved:")
        print(f"  - Detailed results: {results_file}")
        print(f"  - Analysis: {analysis_file}")
        print(f"  - Plots: evaluation_plots_{timestamp}.png")
    
    def _generate_evaluation_plots(self, results: Dict, analysis: Dict, timestamp: str):
        """Generate comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[1, 2] = fig.add_subplot(2, 3, 6, polar=True)

        fig.suptitle('RL Agent Performance Evaluation', fontsize=16)
        
        baselines = list(results.keys())
        
        # 1. Win rates comparison
        win_rates = [results[baseline]['rl_wins'] / len(results[baseline]['rl_scores']) 
                    for baseline in baselines]
        
        axes[0, 0].bar(baselines, win_rates, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Win Rates vs Different Baselines')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, v in enumerate(win_rates):
            axes[0, 0].text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        # 2. Score differences distribution
        all_score_diffs = []
        labels = []
        for baseline in baselines:
            all_score_diffs.extend(results[baseline]['score_differences'])
            labels.extend([baseline] * len(results[baseline]['score_differences']))
        
        df_scores = pd.DataFrame({'Baseline': labels, 'Score_Difference': all_score_diffs})
        sns.boxplot(data=df_scores, x='Baseline', y='Score_Difference', ax=axes[0, 1])
        axes[0, 1].set_title('Score Difference Distribution')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Game length comparison
        avg_game_lengths = [np.mean(results[baseline]['game_lengths']) for baseline in baselines]
        axes[0, 2].bar(baselines, avg_game_lengths, color='lightgreen', alpha=0.7)
        axes[0, 2].set_title('Average Game Length')
        axes[0, 2].set_ylabel('Number of Turns')
        
        # 4. Performance trends over time (if multiple games)
        if len(results[baselines[0]]['score_differences']) > 10:
            for baseline in baselines:
                score_diffs = results[baseline]['score_differences']
                # Calculate moving average
                window_size = max(5, len(score_diffs) // 10)
                moving_avg = np.convolve(score_diffs, np.ones(window_size)/window_size, mode='valid')
                axes[1, 0].plot(moving_avg, label=baseline, alpha=0.7)
            
            axes[1, 0].set_title('Performance Trends (Moving Average)')
            axes[1, 0].set_xlabel('Game Number')
            axes[1, 0].set_ylabel('Score Difference')
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 5. Move time comparison
        rl_move_times = [np.mean(results[baseline]['rl_avg_move_time']) for baseline in baselines]
        baseline_move_times = [np.mean(results[baseline]['baseline_avg_move_time']) for baseline in baselines]
        
        x = np.arange(len(baselines))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, rl_move_times, width, label='RL Agent', alpha=0.7)
        axes[1, 1].bar(x + width/2, baseline_move_times, width, label='Baseline', alpha=0.7)
        axes[1, 1].set_title('Average Move Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(baselines)
        axes[1, 1].legend()
        
        # 6. Performance radar chart
        metrics = ['Win Rate', 'Avg Score Diff', 'Consistency', 'Speed']
        
        # Normalize metrics for radar chart
        normalized_metrics = []
        for baseline in baselines:
            win_rate = results[baseline]['rl_wins'] / len(results[baseline]['rl_scores'])
            avg_score_diff = np.mean(results[baseline]['score_differences'])
            consistency = 1 / (1 + np.std(results[baseline]['score_differences']))  # Higher is better
            speed = 1 / (1 + np.mean(results[baseline]['rl_avg_move_time']))  # Higher is better
            
            normalized_metrics.append([
                win_rate,
                (avg_score_diff + 50) / 100,  # Normalize to 0-1
                consistency,
                speed
            ])
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        axes[1, 2].set_theta_offset(np.pi / 2)
        axes[1, 2].set_theta_direction(-1)
        axes[1, 2].set_thetagrids(np.degrees(angles[:-1]), metrics)
        
        for i, baseline in enumerate(baselines):
            values = normalized_metrics[i] + [normalized_metrics[i][0]]
            axes[1, 2].plot(angles, values, 'o-', linewidth=2, label=baseline, alpha=0.7)
            axes[1, 2].fill(angles, values, alpha=0.1)
        
        axes[1, 2].set_title('Performance Radar Chart')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'evaluation_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def learning_curve_analysis(self, rl_agent: QLearningAgent, 
                               baseline_agent, num_episodes: int = 500,
                               eval_interval: int = 50) -> Dict:
        """Analyze learning curve of RL agent"""
        print(f"Analyzing learning curve over {num_episodes} episodes...")
        
        performance_data = {
            'episodes': [],
            'win_rates': [],
            'avg_scores': [],
            'epsilon_values': [],
            'q_table_sizes': []
        }
        
        for episode in range(0, num_episodes, eval_interval):
            # Evaluate current performance
            eval_results = self._evaluate_head_to_head(
                rl_agent, baseline_agent, 20  # Small evaluation set
            )
            
            win_rate = eval_results['rl_wins'] / 20
            avg_score = np.mean(eval_results['rl_scores'])
            
            performance_data['episodes'].append(episode)
            performance_data['win_rates'].append(win_rate)
            performance_data['avg_scores'].append(avg_score)
            performance_data['epsilon_values'].append(rl_agent.epsilon)
            performance_data['q_table_sizes'].append(len(rl_agent.q_table))
            
            print(f"Episode {episode}: Win Rate = {win_rate:.1%}, "
                  f"Avg Score = {avg_score:.1f}, Epsilon = {rl_agent.epsilon:.3f}")
            
            # Continue training
            for _ in range(eval_interval):
                self._play_training_game(rl_agent, baseline_agent)
        
        return performance_data
    
    def _play_training_game(self, rl_agent: QLearningAgent, baseline_agent):
        """Play a single training game"""
        game = ScrabbleGame(self.dictionary_file)
        game.add_player(rl_agent)
        game.add_player(baseline_agent)
        
        # Initialize opponent modeling
        if hasattr(rl_agent, 'initialize_opponent_model'):
            rl_agent.initialize_opponent_model(game.tile_bag.letter_distribution)
        
        # Play game
        self._play_game_with_metrics(game, rl_agent, baseline_agent)
    
    def ablation_study(self, base_agent: QLearningAgent, 
                      modifications: Dict, num_games: int = 100) -> Dict:
        """Perform ablation study on RL agent components"""
        print("Performing ablation study...")
        
        results = {}
        baseline_agent = GreedyAgent("Greedy_Baseline")
        
        for mod_name, mod_params in modifications.items():
            print(f"Testing modification: {mod_name}")
            
            # Create modified agent
            modified_agent = QLearningAgent(f"RL_{mod_name}")
            
            # Apply modifications
            for param, value in mod_params.items():
                setattr(modified_agent, param, value)
            
            # Evaluate
            eval_results = self._evaluate_head_to_head(
                modified_agent, baseline_agent, num_games
            )
            
            results[mod_name] = {
                'win_rate': eval_results['rl_wins'] / num_games,
                'avg_score_diff': np.mean(eval_results['score_differences']),
                'parameters': mod_params
            }
            
            print(f"  Win Rate: {results[mod_name]['win_rate']:.1%}")
        
        return results