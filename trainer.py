"""
Self-Play Training System for Scrabble RL Agent
Train RL agent against itself with periodic greedy evaluation
"""

import random
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import copy

from scrabble_agent import GreedyAgent
from scrabble_agent import AdaptiveScrabbleQLearner
from move_generator import MoveGenerator
from utils import (create_empty_board, create_tile_bag, draw_tiles, 
                  place_word_on_board, get_rack_after_move, create_game_state,
                  save_game_data, format_time)

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class SelfPlayTrainer:
    """
    Advanced trainer that supports self-play training with periodic evaluation
    """
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        self.move_generator = MoveGenerator(dictionary_path)
        # Enhanced training statistics for self-play
        self.training_stats = {
            'episodes_completed': 0,
            'total_training_time': 0,
            'training_mode': '',  # 'self_play' or 'vs_greedy'
            
            # Self-play training stats
            'self_play_scores_p1': [],     # Player 1 (main agent) scores
            'self_play_scores_p2': [],     # Player 2 (opponent agent) scores  
            'self_play_score_gaps': [],    # P1 - P2 score gaps
            'self_play_wins': [],          # Who won each game (1 or 2)
            
            # Greedy evaluation stats
            'greedy_eval_episodes': [],    # Which episodes had greedy eval
            'greedy_eval_results': [],     # Evaluation results
            'greedy_win_rates': [],        # Win rate vs greedy over time
            'greedy_avg_scores': [],       # Average score vs greedy
            'greedy_score_gaps': [],       # Score gaps vs greedy
            
            # Technical stats
            'weight_evolution': [],
            'network_analysis': [],
            'buffer_statistics': [],
            'td_error_evolution': []
        }

        os.makedirs('plot', exist_ok=True)

    def train_self_play(self, agent: AdaptiveScrabbleQLearner, 
                       num_episodes: int = 2000,
                       greedy_eval_interval: int = 1,  # Evaluate vs greedy every episode
                       greedy_eval_games: int = 3,     # 3 games vs greedy per evaluation
                       verbose: bool = True) -> AdaptiveScrabbleQLearner:
        """
        Train agent using self-play with periodic greedy evaluation
        
        Args:
            agent: The RL agent to train
            num_episodes: Number of self-play episodes
            greedy_eval_interval: How often to evaluate vs greedy (1 = every episode)
            greedy_eval_games: Number of games vs greedy per evaluation
            verbose: Print progress updates
        """
        if verbose:
            print(f"🤖 SELF-PLAY TRAINING WITH GREEDY EVALUATION")
            print(f"=" * 60)
            print(f"Training Episodes: {num_episodes} (RL vs RL)")
            print(f"Greedy Evaluation: Every {greedy_eval_interval} episode(s)")
            print(f"Games per Evaluation: {greedy_eval_games}")
            print(f"Experience Replay: Buffer {agent.experience_buffer.max_size}, Batch {agent.batch_size}")
            print(f"Target Network: Update every {agent.target_update_frequency} updates")
            print("=" * 60)
        
        self.training_stats['training_mode'] = 'self_play'
        start_time = time.time()
        greedy_opponent = GreedyAgent()
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # 1. SELF-PLAY TRAINING GAME
            # Create a copy of the agent for opponent (same weights, different exploration)
            opponent_agent = self._create_opponent_agent(agent)
            
            # Play self-play game
            game_result = self._play_self_play_game(agent, opponent_agent)
            
            # Train both agents from the game (they share the same network)
            if game_result['agent_experiences']:
                agent.train_on_episode(game_result['agent_experiences'])
                td_error = agent.get_last_td_error()
                self.training_stats['td_error_evolution'].append(td_error)
            
            # Store self-play results
            self.training_stats['self_play_scores_p1'].append(game_result['agent_score'])
            self.training_stats['self_play_scores_p2'].append(game_result['opponent_score'])
            self.training_stats['self_play_score_gaps'].append(game_result['final_score_gap'])
            self.training_stats['self_play_wins'].append(1 if game_result['agent_won'] else 2)
            
            self.training_stats['episodes_completed'] += 1
            episode_time = time.time() - episode_start_time
            
            # 2. PERIODIC GREEDY EVALUATION
            if (episode + 1) % greedy_eval_interval == 0:
                eval_start_time = time.time()
                
                # Evaluate current agent vs greedy
                greedy_results = self._evaluate_vs_greedy(agent, greedy_opponent, greedy_eval_games)
                
                # Store evaluation results
                self.training_stats['greedy_eval_episodes'].append(episode + 1)
                self.training_stats['greedy_eval_results'].append(greedy_results)
                self.training_stats['greedy_win_rates'].append(greedy_results['win_rate'])
                self.training_stats['greedy_avg_scores'].append(greedy_results['avg_score'])
                self.training_stats['greedy_score_gaps'].append(greedy_results['avg_score_gap'])
                
                eval_time = time.time() - eval_start_time
                
                # Network analysis
                network_analysis = agent.analyze_networks()
                self.training_stats['network_analysis'].append({
                    'episode': episode + 1,
                    **network_analysis
                })
                
                # Buffer statistics
                buffer_stats = {
                    'episode': episode + 1,
                    'buffer_size': agent.experience_buffer.size(),
                    'buffer_utilization': agent.experience_buffer.size() / agent.experience_buffer.max_size,
                    'total_updates': agent.total_updates,
                    'target_updates': agent.target_updates
                }
                self.training_stats['buffer_statistics'].append(buffer_stats)
                
                if verbose:
                    self._print_self_play_progress(
                        episode + 1, game_result, greedy_results, 
                        network_analysis, buffer_stats, episode_time, eval_time
                    )
            elif verbose and (episode + 1) % 50 == 0:
                # Print self-play progress without evaluation
                print(f"Episode {episode + 1:4d} | "
                      f"Self-Play: {game_result['agent_score']:3.0f} vs {game_result['opponent_score']:3.0f} | "
                      f"Gap: {game_result['final_score_gap']:+4.0f} | "
                      f"Time: {format_time(episode_time)}")

        # Generate plots after training
        self.plot_self_play_progress()
        
        total_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_time
        
        if verbose:
            self._print_final_self_play_summary()
        
        return agent

    def _create_opponent_agent(self, main_agent: AdaptiveScrabbleQLearner) -> AdaptiveScrabbleQLearner:
        """
        Create opponent agent that shares weights but has different exploration
        """
        # Create a copy with same architecture
        opponent = AdaptiveScrabbleQLearner(
            num_features=main_agent.num_features,
            learning_rate=main_agent.learning_rate,
            epsilon=main_agent.epsilon * 1.2,  # Slightly more exploration for diversity
            gamma=main_agent.gamma,
            buffer_size=main_agent.experience_buffer.max_size,
            batch_size=main_agent.batch_size,
            target_update_frequency=main_agent.target_update_frequency
        )
        
        # Copy the main agent's weights
        opponent.main_weights = main_agent.main_weights.copy()
        opponent.target_weights = main_agent.target_weights.copy()
        
        # Copy adaptive components
        opponent.adaptive_timing = copy.deepcopy(main_agent.adaptive_timing)
        opponent.adaptive_tiles = copy.deepcopy(main_agent.adaptive_tiles)
        
        return opponent

    def _play_self_play_game(self, agent1: AdaptiveScrabbleQLearner, 
                            agent2: AdaptiveScrabbleQLearner) -> Dict:
        """
        Play a self-play game between two RL agents
        """
        board = create_empty_board()
        tile_bag = create_tile_bag()
        
        agent1_rack = draw_tiles(tile_bag, 7)
        agent2_rack = draw_tiles(tile_bag, 7)
        
        agent1_score = 0
        agent2_score = 0
        rounds_played = 0
        max_rounds = 50
        
        agent1_experiences = []
        agent2_experiences = []  # We could use this for additional training
        
        while len(tile_bag) > 0 and rounds_played < max_rounds:
            # Agent 1's turn
            agent1_state = create_game_state(
                board, agent1_rack, [], agent1_score, agent2_score,
                len(tile_bag), rounds_played
            )
            
            valid_moves1 = self.move_generator.get_valid_moves(
                board, agent1_rack, sampling_rate=0.3
            )
            
            if valid_moves1:
                chosen_move1 = agent1.choose_move(agent1_state, valid_moves1, training=True)
                
                if chosen_move1:
                    old_state1 = agent1_state.copy()
                    board = place_word_on_board(board, chosen_move1['word'], chosen_move1['positions'])
                    agent1_score += chosen_move1['score']
                    
                    tiles_drawn1 = draw_tiles(tile_bag, len(chosen_move1['tiles_used']))
                    agent1_rack = get_rack_after_move(
                        agent1_rack, chosen_move1['tiles_used'], tiles_drawn1
                    )
                    
                    new_state1 = create_game_state(
                        board, agent1_rack, [], agent1_score, agent2_score,
                        len(tile_bag), rounds_played
                    )
                    
                    reward1 = agent1.calculate_reward(old_state1, new_state1, chosen_move1)
                    
                    experience1 = {
                        'state': old_state1,
                        'move': chosen_move1,
                        'reward': reward1,
                        'next_state': new_state1,
                        'terminal': False
                    }
                    agent1_experiences.append(experience1)
            
            # Agent 2's turn (similar structure)
            if len(tile_bag) > 0:
                agent2_state = create_game_state(
                    board, agent2_rack, [], agent2_score, agent1_score,
                    len(tile_bag), rounds_played
                )
                
                valid_moves2 = self.move_generator.get_valid_moves(
                    board, agent2_rack, sampling_rate=0.3
                )
                
                if valid_moves2:
                    chosen_move2 = agent2.choose_move(agent2_state, valid_moves2, training=True)
                    
                    if chosen_move2:
                        board = place_word_on_board(board, chosen_move2['word'], chosen_move2['positions'])
                        agent2_score += chosen_move2['score']
                        
                        tiles_drawn2 = draw_tiles(tile_bag, len(chosen_move2['tiles_used']))
                        agent2_rack = get_rack_after_move(
                            agent2_rack, chosen_move2['tiles_used'], tiles_drawn2
                        )
            
            rounds_played += 1
            
            if not valid_moves1 and not valid_moves2:
                break
        
        # Mark final experience as terminal
        if agent1_experiences:
            agent1_experiences[-1]['terminal'] = True
        
        final_score_gap = agent1_score - agent2_score
        agent1_won = final_score_gap > 0
        
        return {
            'agent_score': agent1_score,
            'opponent_score': agent2_score,
            'final_score_gap': final_score_gap,
            'agent_won': agent1_won,
            'rounds_played': rounds_played,
            'agent_experiences': agent1_experiences,
            'opponent_experiences': agent2_experiences
        }

    def _evaluate_vs_greedy(self, agent: AdaptiveScrabbleQLearner, 
                           greedy_opponent: GreedyAgent, num_games: int) -> Dict:
        """
        Evaluate agent performance against greedy opponent
        """
        results = {
            'games_played': 0,
            'wins': 0,
            'total_score': 0,
            'total_opponent_score': 0,
            'total_score_gap': 0,
            'win_rate': 0.0,
            'avg_score': 0.0,
            'avg_opponent_score': 0.0,
            'avg_score_gap': 0.0
        }
        
        for _ in range(num_games):
            game_result = self._play_evaluation_game(agent, greedy_opponent)
            
            results['games_played'] += 1
            if game_result['agent_won']:
                results['wins'] += 1
            
            results['total_score'] += game_result['agent_score']
            results['total_opponent_score'] += game_result['opponent_score']
            results['total_score_gap'] += game_result['final_score_gap']
        
        # Calculate averages
        if results['games_played'] > 0:
            results['win_rate'] = results['wins'] / results['games_played']
            results['avg_score'] = results['total_score'] / results['games_played']
            results['avg_opponent_score'] = results['total_opponent_score'] / results['games_played']
            results['avg_score_gap'] = results['total_score_gap'] / results['games_played']
        
        return results

    def _play_evaluation_game(self, agent: AdaptiveScrabbleQLearner, opponent) -> Dict:
        """Play evaluation game (no training updates)"""
        board = create_empty_board()
        tile_bag = create_tile_bag()
        
        agent_rack = draw_tiles(tile_bag, 7)
        opponent_rack = draw_tiles(tile_bag, 7)
        
        agent_score = 0
        opponent_score = 0
        rounds_played = 0
        max_rounds = 50
        
        while len(tile_bag) > 0 and rounds_played < max_rounds:
            # Agent's turn
            agent_state = create_game_state(
                board, agent_rack, [], agent_score, opponent_score,
                len(tile_bag), rounds_played
            )
            
            valid_moves = self.move_generator.get_valid_moves(board, agent_rack)
            
            if valid_moves:
                chosen_move = agent.choose_move(agent_state, valid_moves, training=False)
                
                if chosen_move:
                    board = place_word_on_board(board, chosen_move['word'], chosen_move['positions'])
                    agent_score += chosen_move['score']
                    
                    tiles_drawn = draw_tiles(tile_bag, len(chosen_move['tiles_used']))
                    agent_rack = get_rack_after_move(
                        agent_rack, chosen_move['tiles_used'], tiles_drawn
                    )
            
            # Opponent's turn
            if len(tile_bag) > 0:
                opponent_state = create_game_state(
                    board, opponent_rack, [], opponent_score, agent_score,
                    len(tile_bag), rounds_played
                )
                
                opponent_moves = self.move_generator.get_valid_moves(board, opponent_rack)
                
                if opponent_moves:
                    opponent_move = opponent.choose_move(opponent_state, opponent_moves, training=False)
                    
                    if opponent_move:
                        board = place_word_on_board(
                            board, opponent_move['word'], opponent_move['positions']
                        )
                        opponent_score += opponent_move['score']
                        
                        opponent_tiles_drawn = draw_tiles(tile_bag, len(opponent_move['tiles_used']))
                        opponent_rack = get_rack_after_move(
                            opponent_rack, opponent_move['tiles_used'], opponent_tiles_drawn
                        )
            
            rounds_played += 1
            
            if not valid_moves and not opponent_moves:
                break
        
        return {
            'agent_score': agent_score,
            'opponent_score': opponent_score,
            'final_score_gap': agent_score - opponent_score,
            'agent_won': agent_score > opponent_score,
            'rounds_played': rounds_played
        }

    def _print_self_play_progress(self, episode: int, game_result: Dict, greedy_results: Dict,
                                 network_analysis: Dict, buffer_stats: Dict, 
                                 episode_time: float, eval_time: float):
        """Print progress for self-play training"""
        print(f"Episode {episode:4d} | "
              f"Self-Play: {game_result['agent_score']:3.0f} vs {game_result['opponent_score']:3.0f} | "
              f"Gap: {game_result['final_score_gap']:+4.0f}")
        
        print(f"             | "
              f"vs Greedy: {greedy_results['win_rate']:5.1%} WR | "
              f"Score: {greedy_results['avg_score']:5.1f} | "
              f"Gap: {greedy_results['avg_score_gap']:+5.1f}")
        
        print(f"             | "
              f"Buffer: {buffer_stats['buffer_size']:4d}/{buffer_stats['buffer_utilization']*100:3.0f}% | "
              f"Updates: {buffer_stats['target_updates']:3d} | "
              f"Time: {format_time(episode_time + eval_time)}")

   

    def plot_self_play_progress(self):
        """Generate comprehensive self-play training plots"""
        
        if not self.training_stats['self_play_scores_p1']:
            print("No self-play data to plot")
            return
        
        plt.figure(figsize=(20, 15))
        
        episodes = np.arange(1, len(self.training_stats['self_play_scores_p1']) + 1)
        
        # Plot 1: Self-Play Score Evolution
        plt.subplot(3, 3, 1)
        p1_scores = self.training_stats['self_play_scores_p1']
        p2_scores = self.training_stats['self_play_scores_p2']
        
        plt.plot(episodes, p1_scores, alpha=0.6, label='Agent 1 (Main)', color='blue')
        plt.plot(episodes, p2_scores, alpha=0.6, label='Agent 2 (Copy)', color='red')
        
        # Moving averages
        window = max(1, len(episodes) // 20)
        if len(episodes) >= window:
            p1_ma = np.convolve(p1_scores, np.ones(window)/window, mode='valid')
            p2_ma = np.convolve(p2_scores, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], p1_ma, label='Agent 1 MA', linestyle='--', color='darkblue')
            plt.plot(episodes[window-1:], p2_ma, label='Agent 2 MA', linestyle='--', color='darkred')
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Self-Play Score Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Self-Play Score Gaps
        plt.subplot(3, 3, 2)
        score_gaps = self.training_stats['self_play_score_gaps']
        plt.plot(episodes, score_gaps, alpha=0.6, color='purple')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.fill_between(episodes, score_gaps, 0, where=(np.array(score_gaps) >= 0), 
                        color='green', alpha=0.3, label='Agent 1 Wins')
        plt.fill_between(episodes, score_gaps, 0, where=(np.array(score_gaps) < 0), 
                        color='red', alpha=0.3, label='Agent 2 Wins')
        plt.xlabel('Episode')
        plt.ylabel('Score Gap (Agent 1 - Agent 2)')
        plt.title('Self-Play Score Gaps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # plot 3
        plt.subplot(3, 3, 3)
        if self.training_stats['greedy_eval_episodes']:
            eval_episodes = self.training_stats['greedy_eval_episodes']
            win_rates = [wr * 100 for wr in self.training_stats['greedy_win_rates']]
            
            plt.scatter(eval_episodes, win_rates, s=10, color='lightgreen', label='Raw Win Rate')
            smoothed_win_rate = moving_average(win_rates, window_size=10)
            plt.plot(eval_episodes[:len(smoothed_win_rate)], smoothed_win_rate, 
                    linewidth=2.5, color='green', label='Smoothed Win Rate')
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% (Even)')
            
            plt.xlabel('Episode')
            plt.ylabel('Win Rate vs Greedy (%)')
            plt.title('Performance vs Greedy Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)

        # Plot 4: Score vs Greedy Over Time
        plt.subplot(3, 3, 4)
        if self.training_stats['greedy_eval_episodes']:
            eval_episodes = self.training_stats['greedy_eval_episodes']
            agent_scores = self.training_stats['greedy_avg_scores']
            
            plt.scatter(eval_episodes, agent_scores, s=10, color='lightblue', label='Raw Scores')
            smoothed_scores = moving_average(agent_scores, window_size=10)
            plt.plot(eval_episodes[:len(smoothed_scores)], smoothed_scores, 
                    linewidth=2.5, color='blue', label='Smoothed Score')
            
            plt.xlabel('Episode')
            plt.ylabel('Average Score vs Greedy')
            plt.title('Score Performance vs Greedy')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot 6: Score Gap vs Greedy Over Time
        plt.subplot(3, 3, 5)
        if self.training_stats['greedy_score_gaps']:
            eval_episodes = self.training_stats['greedy_eval_episodes']
            score_gaps = self.training_stats['greedy_score_gaps']
            
            plt.scatter(eval_episodes, score_gaps, s=10, color='navajowhite', label='Raw Score Gap')
            smoothed_gaps = moving_average(score_gaps, window_size=10)
            plt.plot(eval_episodes[:len(smoothed_gaps)], smoothed_gaps, 
                    linewidth=2.5, color='orange', label='Smoothed Score Gap')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.xlabel('Episode')
            plt.ylabel('Score Gap vs Greedy')
            plt.title('Score Gap vs Greedy Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 7: Score Distribution Comparison
        plt.subplot(3, 3, 6)
        plt.hist(p1_scores, bins=30, alpha=0.7, label='Agent 1', color='blue', density=True)
        plt.hist(p2_scores, bins=30, alpha=0.7, label='Agent 2', color='red', density=True)
        plt.axvline(np.mean(p1_scores), color='darkblue', linestyle='--', 
                   label=f'Agent 1 Avg: {np.mean(p1_scores):.1f}')
        plt.axvline(np.mean(p2_scores), color='darkred', linestyle='--', 
                   label=f'Agent 2 Avg: {np.mean(p2_scores):.1f}')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.title('Self-Play Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 8: TD Error Evolution
        plt.subplot(3, 3, 7)
        if self.training_stats['td_error_evolution']:
            td_errors = self.training_stats['td_error_evolution']
            td_episodes = np.arange(1, len(td_errors) + 1)
            
            plt.plot(td_episodes, td_errors, alpha=0.7, color='orange')
            if len(td_errors) > 10:
                td_window = max(1, len(td_errors) // 20)
                td_ma = np.convolve(td_errors, np.ones(td_window)/td_window, mode='valid')
                plt.plot(td_episodes[td_window-1:], td_ma, linestyle='--', linewidth=2, color='darkorange')
            
            plt.xlabel('Episode')
            plt.ylabel('TD Error')
            plt.title('TD Error Evolution')
            plt.grid(True, alpha=0.3)

        
        
        # Plot 11: Network Analysis
        plt.subplot(3, 3, 8)
        if self.training_stats['network_analysis']:
            net_episodes = [entry['episode'] for entry in self.training_stats['network_analysis']]
            weight_diffs = [entry['weight_difference_norm'] for entry in self.training_stats['network_analysis']]
            
            plt.plot(net_episodes, weight_diffs, marker='^', linewidth=2, 
                    markersize=4, color='purple', label='Main-Target Weight Diff')
            plt.xlabel('Episode')
            plt.ylabel('Weight Difference Norm')
            plt.title('Network Divergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 12: Buffer Utilization
        plt.subplot(3, 3, 9)
        if self.training_stats['buffer_statistics']:
            buf_episodes = [entry['episode'] for entry in self.training_stats['buffer_statistics']]
            buf_utilization = [entry['buffer_utilization'] * 100 for entry in self.training_stats['buffer_statistics']]
            
            plt.plot(buf_episodes, buf_utilization, marker='x', linewidth=2, 
                    markersize=4, color='brown', label='Buffer Utilization')
            plt.xlabel('Episode')
            plt.ylabel('Buffer Utilization (%)')
            plt.title('Experience Replay Buffer Usage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 105)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plot/self_play_training_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Self-play training plots saved: plot/self_play_training_{timestamp}.png")

    def _print_final_self_play_summary(self):
        """Print comprehensive final summary"""
        print(f"\n" + "="*70)
        print(f"🤖 SELF-PLAY TRAINING COMPLETED")
        print(f"="*70)
        
        episodes = len(self.training_stats['self_play_scores_p1'])
        p1_avg = np.mean(self.training_stats['self_play_scores_p1'])
        p2_avg = np.mean(self.training_stats['self_play_scores_p2'])
        
        print(f"Total Self-Play Episodes: {episodes}")
        print(f"Agent 1 Average Score: {p1_avg:.1f}")
        print(f"Agent 2 Average Score: {p2_avg:.1f}")
        
        # Self-play balance
        wins = self.training_stats['self_play_wins']
        agent1_wins = sum(1 for w in wins if w == 1)
        agent1_win_rate = agent1_wins / len(wins) * 100
        
        print(f"Agent 1 Win Rate in Self-Play: {agent1_win_rate:.1f}%")
        
        if 45 <= agent1_win_rate <= 55:
            print("✅ Self-play is well balanced!")
        elif agent1_win_rate > 60:
            print("⚠️ Agent 1 dominates - consider adjusting exploration")
        else:
            print("⚠️ Agent 2 dominates - this is unusual")
        
        # Greedy evaluation summary
        if self.training_stats['greedy_win_rates']:
            print(f"\n🎯 PERFORMANCE VS GREEDY:")
            
            early_wr = np.mean(self.training_stats['greedy_win_rates'][:5]) if len(self.training_stats['greedy_win_rates']) >= 5 else 0
            recent_wr = np.mean(self.training_stats['greedy_win_rates'][-5:])
            improvement = recent_wr - early_wr
            
            print(f"Early Win Rate vs Greedy: {early_wr:.1%}")
            print(f"Final Win Rate vs Greedy: {recent_wr:.1%}")
            print(f"Improvement: {improvement:+.1%}")
            
            final_score = np.mean(self.training_stats['greedy_avg_scores'][-5:])
            final_gap = np.mean(self.training_stats['greedy_score_gaps'][-5:])
            
            print(f"Final Average Score vs Greedy: {final_score:.1f}")
            print(f"Final Average Score Gap: {final_gap:+.1f}")
            
            if recent_wr > 0.7:
                print("🚀 EXCELLENT - Agent dominates greedy opponent!")
            elif recent_wr > 0.6:
                print("✅ GOOD - Agent consistently beats greedy")
            elif recent_wr > 0.5:
                print("📊 FAIR - Agent is competitive with greedy")
            else:
                print("⚠️ NEEDS WORK - Agent struggles against greedy")
            
            if improvement > 0.2:
                print("📈 Strong learning curve from self-play!")
            elif improvement > 0.1:
                print("📈 Good improvement from self-play")
            elif improvement > 0:
                print("📊 Modest improvement from self-play")
            else:
                print("📉 No clear improvement - may need more episodes")
        
        print(f"="*70)

    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        return {
            'training_stats': self.training_stats,
            'training_mode': 'self_play',
            'episodes_completed': self.training_stats['episodes_completed'],
            'total_training_time': self.training_stats['total_training_time']
        }

    def save_training_data(self, filepath: str):
        """Save self-play training statistics"""
        training_data = {
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'self_play',
            'version': '1.0_self_play_with_greedy_eval'
        }
        save_game_data(training_data, filepath)


# Enhanced main.py functions for self-play
def train_self_play_agent(args):
    """Train agent using self-play with greedy evaluation"""
    print("🤖 SELF-PLAY TRAINING MODE")
    print("=" * 50)
    print(f"Episodes: {args.episodes} (RL vs RL)")
    print(f"Greedy Evaluation: Every {args.greedy_eval_interval} episode(s)")
    print(f"Games per Evaluation: {args.greedy_eval_games}")
    print(f"Learning Rate: {args.learning_rate}")
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


def main_with_self_play():
    """Enhanced main function with self-play support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrabble RL Agent with Self-Play Training')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'self-play':
        train_self_play_agent(args)


if __name__ == "__main__":
    main_with_self_play()