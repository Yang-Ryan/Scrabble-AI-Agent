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

from scrabble_agent import GreedyAgent, AdaptiveScrabbleQLearner, QuackleAgent
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

    def _train_vs_greedy(self, agent: AdaptiveScrabbleQLearner, num_episodes: int,
                     evaluation_interval: int = 100, save_interval: int = 500, verbose: bool = True) -> AdaptiveScrabbleQLearner:
        """
        Train the agent against a GreedyAgent opponent.
        """
        greedy_opponent = GreedyAgent()
        self.training_logs = []

        for episode in range(1, num_episodes + 1):
            board = create_empty_board()
            tile_bag = create_tile_bag()
            
            agent_rack = draw_tiles(tile_bag, 7)
            opponent_rack = draw_tiles(tile_bag, 7)
            
            agent_score = 0
            opponent_score = 0
            rounds_played = 0
            max_rounds = 50

            agent_experiences = []

            while len(tile_bag) > 0 and rounds_played < max_rounds:
                # Agent's turn
                agent_state = create_game_state(
                    board, agent_rack, [], agent_score, opponent_score,
                    len(tile_bag), rounds_played
                )
                valid_moves = self.move_generator.get_valid_moves(board, agent_rack)
                
                if valid_moves:
                    chosen_move = agent.choose_move(agent_state, valid_moves, training=True)
                    if chosen_move:
                        old_state = agent_state.copy()

                        board = place_word_on_board(board, chosen_move['word'], chosen_move['positions'])
                        agent_score += chosen_move['score']
                        
                        tiles_drawn = draw_tiles(tile_bag, len(chosen_move['tiles_used']))
                        agent_rack = get_rack_after_move(agent_rack, chosen_move['tiles_used'], tiles_drawn)
                        
                        new_state = create_game_state(
                            board, agent_rack, [], agent_score, opponent_score,
                            len(tile_bag), rounds_played
                        )
                        
                        reward = agent.calculate_reward(old_state, new_state, chosen_move)

                        # Record experience
                        experience = {
                            'state': old_state,
                            'move': chosen_move,
                            'reward': reward,
                            'next_state': new_state,
                            'terminal': False
                        }
                        agent_experiences.append(experience)

                # Opponent's turn
                if len(tile_bag) > 0:
                    opponent_state = create_game_state(
                        board, opponent_rack, [], opponent_score, agent_score,
                        len(tile_bag), rounds_played
                    )
                    opponent_moves = self.move_generator.get_valid_moves(board, opponent_rack)
                    
                    if opponent_moves:
                        opponent_move = greedy_opponent.choose_move(opponent_state, opponent_moves, training=False)
                        if opponent_move:
                            board = place_word_on_board(board, opponent_move['word'], opponent_move['positions'])
                            opponent_score += opponent_move['score']
                            
                            opponent_tiles_drawn = draw_tiles(tile_bag, len(opponent_move['tiles_used']))
                            opponent_rack = get_rack_after_move(opponent_rack, opponent_move['tiles_used'], opponent_tiles_drawn)

                rounds_played += 1

                if not valid_moves and not opponent_moves:
                    break

            if agent_experiences:
                # Mark last move as terminal
                agent_experiences[-1]['terminal'] = True

                # Train after the game
                agent.train_on_episode(agent_experiences)
            
            self.training_logs.append({
                'episode': episode,
                'agent_score': agent_score,
                'opponent_score': opponent_score,
                'final_score_gap': agent_score - opponent_score,
                'agent_won': agent_score > opponent_score
            })

            if evaluation_interval > 0 and episode % evaluation_interval == 0:
                wins = sum(1 for log in self.training_logs[-evaluation_interval:] if log['agent_won'])
                avg_score = sum(log['agent_score'] for log in self.training_logs[-evaluation_interval:]) / evaluation_interval
                avg_gap = sum(log['final_score_gap'] for log in self.training_logs[-evaluation_interval:]) / evaluation_interval

                if verbose:
                    print(f"[Episode {episode}] Win Rate: {wins / evaluation_interval:.1%}, "
                        f"Avg Score: {avg_score:.1f}, Avg Gap: {avg_gap:+.1f}")

            if save_interval > 0 and episode % save_interval == 0:
                checkpoint_path = f"checkpoint_episode_{episode}.json"
                agent.save_model(checkpoint_path)
                if verbose:
                    print(f"Model checkpoint saved: {checkpoint_path}")
        
        final_wins = sum(1 for log in self.training_logs if log['agent_won'])
        games_played = len(self.training_logs)
        self.training_summary = {
            'final_performance': {
                'win_rate': final_wins / games_played if games_played > 0 else 0,
                'avg_score': sum(log['agent_score'] for log in self.training_logs) / games_played if games_played > 0 else 0,
                'avg_score_gap': sum(log['final_score_gap'] for log in self.training_logs) / games_played if games_played > 0 else 0
            }
        }
        
        return agent


    def _evaluate_vs_greedy(self, agent: AdaptiveScrabbleQLearner, 
                        greedy_opponent: GreedyAgent, num_games: int) -> Dict:
        """
        Evaluate agent performance against greedy opponent (no training updates)
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
                        agent_rack = get_rack_after_move(agent_rack, chosen_move['tiles_used'], tiles_drawn)
                
                # Opponent's turn
                if len(tile_bag) > 0:
                    opponent_state = create_game_state(
                        board, opponent_rack, [], opponent_score, agent_score,
                        len(tile_bag), rounds_played
                    )
                    opponent_moves = self.move_generator.get_valid_moves(board, opponent_rack)
                    
                    if opponent_moves:
                        opponent_move = greedy_opponent.choose_move(opponent_state, opponent_moves, training=False)
                        if opponent_move:
                            board = place_word_on_board(board, opponent_move['word'], opponent_move['positions'])
                            opponent_score += opponent_move['score']
                            
                            opponent_tiles_drawn = draw_tiles(tile_bag, len(opponent_move['tiles_used']))
                            opponent_rack = get_rack_after_move(opponent_rack, opponent_move['tiles_used'], opponent_tiles_drawn)
                
                rounds_played += 1
                
                if not valid_moves and not opponent_moves:
                    break
            
            # Record results
            results['games_played'] += 1
            if agent_score > opponent_score:
                results['wins'] += 1
            
            results['total_score'] += agent_score
            results['total_opponent_score'] += opponent_score
            results['total_score_gap'] += (agent_score - opponent_score)
        
        # Calculate averages
        if results['games_played'] > 0:
            results['win_rate'] = results['wins'] / results['games_played']
            results['avg_score'] = results['total_score'] / results['games_played']
            results['avg_opponent_score'] = results['total_opponent_score'] / results['games_played']
            results['avg_score_gap'] = results['total_score_gap'] / results['games_played']
        
        return results


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

class QuackleTrainer:
    """
    Trainer class specifically for training RL agents against Quackle opponents
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        self.move_generator = MoveGenerator(dictionary_path)
        self.training_stats = {
            'training_mode': 'vs_quackle',
            'episodes_completed': 0,
            'total_training_time': 0,
            'agent_scores': [],
            'quackle_scores': [],
            'score_gaps': [],
            'wins': [],
            'td_errors': [],
            'weight_evolution': [],
            'evaluation_results': []
        }
        
        # Ensure plot directory exists
        os.makedirs('plot', exist_ok=True)
    
    def train_vs_quackle(self, agent: AdaptiveScrabbleQLearner, num_episodes: int,
                        evaluation_interval: int = 25, save_interval: int = 250, 
                        verbose: bool = True) -> AdaptiveScrabbleQLearner:
        """
        Train the agent against Quackle opponent with comprehensive tracking.
        """
        from scrabble_agent import QuackleAgent
        
        if verbose:
            print("🦆 QUACKLE TRAINING STARTED")
            print("=" * 50)
            print(f"Training Episodes: {num_episodes}")
            print(f"Evaluation Interval: {evaluation_interval}")
            print(f"Experience Replay: Buffer {agent.experience_buffer.max_size}, Batch {agent.batch_size}")
            print("=" * 50)
        
        # Initialize Quackle opponent
        quackle_opponent = QuackleAgent()
        
        # Reset training stats for this session
        self._reset_training_stats()
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            episode_start_time = time.time()
            
            # Play one training game
            game_result = self._play_vs_quackle_game(agent, quackle_opponent)
            
            # Train agent from the game experience
            if game_result['agent_experiences']:
                agent.train_on_episode(game_result['agent_experiences'])
                td_error = agent.get_last_td_error()
                self.training_stats['td_errors'].append(td_error)
            
            # Store game results
            self.training_stats['agent_scores'].append(game_result['agent_score'])
            self.training_stats['quackle_scores'].append(game_result['quackle_score'])
            self.training_stats['score_gaps'].append(game_result['final_score_gap'])
            self.training_stats['wins'].append(game_result['agent_won'])
            self.training_stats['episodes_completed'] += 1
            
            episode_time = time.time() - episode_start_time
            
            # Periodic evaluation and progress reporting
            if episode % evaluation_interval == 0:
                recent_games = evaluation_interval
                recent_wins = sum(self.training_stats['wins'][-recent_games:])
                recent_avg_score = np.mean(self.training_stats['agent_scores'][-recent_games:])
                recent_avg_gap = np.mean(self.training_stats['score_gaps'][-recent_games:])
                win_rate = recent_wins / recent_games
                
                # Store evaluation
                eval_result = {
                    'episode': episode,
                    'win_rate': win_rate,
                    'avg_score': recent_avg_score,
                    'avg_score_gap': recent_avg_gap,
                    'epsilon': agent.epsilon,
                    'buffer_size': agent.experience_buffer.size()
                }
                self.training_stats['evaluation_results'].append(eval_result)
                
                if verbose:
                    print(f"[Episode {episode:4d}] "
                          f"vs Quackle - WR: {win_rate:5.1%} | "
                          f"Score: {recent_avg_score:5.1f} | "
                          f"Gap: {recent_avg_gap:+5.1f} | "
                          f"ε: {agent.epsilon:.3f} | "
                          f"Buffer: {agent.experience_buffer.size():4d} | "
                          f"Time: {format_time(episode_time)}")
            
            # Save checkpoints
            if save_interval > 0 and episode % save_interval == 0:
                checkpoint_path = f"quackle_checkpoint_ep{episode}.json"
                agent.save_model(checkpoint_path)
                if verbose:
                    print(f"    Checkpoint saved: {checkpoint_path}")
        
        total_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_time
        
        # Generate final summary
        if verbose:
            self._print_quackle_training_summary()
        
        # Generate plots
        self._plot_quackle_training_progress()
        
        return agent
    
    def _reset_training_stats(self):
        """Reset training statistics for a new training session"""
        self.training_stats = {
            'training_mode': 'vs_quackle',
            'episodes_completed': 0,
            'total_training_time': 0,
            'agent_scores': [],
            'quackle_scores': [],
            'score_gaps': [],
            'wins': [],
            'td_errors': [],
            'weight_evolution': [],
            'evaluation_results': []
        }
    
    def _play_vs_quackle_game(self, agent: AdaptiveScrabbleQLearner, 
                             quackle_opponent) -> Dict:
        """
        Play a single game between RL agent and Quackle opponent
        """
        board = create_empty_board()
        tile_bag = create_tile_bag()
        
        agent_rack = draw_tiles(tile_bag, 7)
        quackle_rack = draw_tiles(tile_bag, 7)
        
        agent_score = 0
        quackle_score = 0
        rounds_played = 0
        max_rounds = 50
        
        agent_experiences = []
        
        while len(tile_bag) > 0 and rounds_played < max_rounds:
            # Agent's turn
            agent_state = create_game_state(
                board, agent_rack, [], agent_score, quackle_score,
                len(tile_bag), rounds_played
            )
            
            valid_moves = self.move_generator.get_valid_moves(
                board, agent_rack, sampling_rate=0.5  # Higher sampling for Quackle training
            )
            
            if valid_moves:
                chosen_move = agent.choose_move(agent_state, valid_moves, training=True)
                
                if chosen_move:
                    old_state = agent_state.copy()
                    
                    # Apply agent's move
                    board = place_word_on_board(board, chosen_move['word'], chosen_move['positions'])
                    agent_score += chosen_move['score']
                    
                    # Update agent's rack
                    tiles_drawn = draw_tiles(tile_bag, len(chosen_move['tiles_used']))
                    agent_rack = get_rack_after_move(
                        agent_rack, chosen_move['tiles_used'], tiles_drawn
                    )
                    
                    # Create new state for reward calculation
                    new_state = create_game_state(
                        board, agent_rack, [], agent_score, quackle_score,
                        len(tile_bag), rounds_played
                    )
                    
                    # Calculate reward
                    reward = agent.calculate_reward(old_state, new_state, chosen_move)
                    
                    # Store experience
                    experience = {
                        'state': old_state,
                        'move': chosen_move,
                        'reward': reward,
                        'next_state': new_state,
                        'terminal': False
                    }
                    agent_experiences.append(experience)
                    
                    # Record move for Quackle (if needed for GCG tracking)
                    try:
                        quackle_opponent.agent.to_quackle_input("us", chosen_move)
                    except Exception as e:
                        # Handle GCG recording errors gracefully
                        if "file" not in str(e).lower():
                            print(f"Warning: GCG recording issue: {e}")
            
            # Quackle's turn
            if len(tile_bag) > 0:
                quackle_state = create_game_state(
                    board, quackle_rack, [], quackle_score, agent_score,
                    len(tile_bag), rounds_played
                )
                
                quackle_moves = self.move_generator.get_valid_moves(
                    board, quackle_rack, sampling_rate=0.5
                )
                
                if quackle_moves:
                    try:
                        quackle_move = quackle_opponent.choose_move(
                            quackle_state, quackle_moves, training=False
                        )
                        
                        if quackle_move:
                            # Apply Quackle's move
                            board = place_word_on_board(
                                board, quackle_move['word'], quackle_move['positions']
                            )
                            quackle_score += quackle_move['score']
                            
                            # Update Quackle's rack
                            quackle_tiles_drawn = draw_tiles(tile_bag, len(quackle_move['tiles_used']))
                            quackle_rack = get_rack_after_move(
                                quackle_rack, quackle_move['tiles_used'], quackle_tiles_drawn
                            )
                            
                            # Record Quackle's move
                            try:
                                quackle_opponent.agent.to_quackle_input("quackle", quackle_move)
                            except Exception as e:
                                if "file" not in str(e).lower():
                                    print(f"Warning: Quackle GCG recording issue: {e}")
                        
                    except Exception as e:
                        print(f"Warning: Quackle move generation failed: {e}")
                        # Fallback to random valid move
                        if quackle_moves:
                            quackle_move = random.choice(quackle_moves)
                            board = place_word_on_board(
                                board, quackle_move['word'], quackle_move['positions']
                            )
                            quackle_score += quackle_move['score']
                            
                            quackle_tiles_drawn = draw_tiles(tile_bag, len(quackle_move['tiles_used']))
                            quackle_rack = get_rack_after_move(
                                quackle_rack, quackle_move['tiles_used'], quackle_tiles_drawn
                            )
            
            rounds_played += 1
            
            # Break if no valid moves for both players
            if not valid_moves and not quackle_moves:
                break
        
        # Mark final experience as terminal
        if agent_experiences:
            agent_experiences[-1]['terminal'] = True
        
        final_score_gap = agent_score - quackle_score
        agent_won = final_score_gap > 0
        
        return {
            'agent_score': agent_score,
            'quackle_score': quackle_score,
            'final_score_gap': final_score_gap,
            'agent_won': agent_won,
            'rounds_played': rounds_played,
            'agent_experiences': agent_experiences
        }
    
    def _print_quackle_training_summary(self):
        """Print comprehensive Quackle training summary"""
        print(f"\n" + "="*60)
        print(f"🦆 QUACKLE TRAINING COMPLETED")
        print(f"="*60)
        
        episodes = len(self.training_stats['agent_scores'])
        total_wins = sum(self.training_stats['wins'])
        win_rate = total_wins / episodes if episodes > 0 else 0
        
        avg_agent_score = np.mean(self.training_stats['agent_scores'])
        avg_quackle_score = np.mean(self.training_stats['quackle_scores'])
        avg_score_gap = np.mean(self.training_stats['score_gaps'])
        
        print(f"Total Episodes: {episodes}")
        print(f"Overall Win Rate vs Quackle: {win_rate:.1%}")
        print(f"Average Agent Score: {avg_agent_score:.1f}")
        print(f"Average Quackle Score: {avg_quackle_score:.1f}")
        print(f"Average Score Gap: {avg_score_gap:+.1f}")
        
        # Performance assessment
        if win_rate > 0.6:
            print("🚀 EXCELLENT - Agent performs well against Quackle!")
        elif win_rate > 0.5:
            print("✅ GOOD - Agent is competitive with Quackle")
        elif win_rate > 0.4:
            print("📊 FAIR - Agent shows promise against Quackle")
        else:
            print("⚠️ NEEDS WORK - Agent struggles against Quackle")
        
        # Learning progression
        if len(self.training_stats['evaluation_results']) >= 2:
            early_wr = self.training_stats['evaluation_results'][0]['win_rate']
            final_wr = self.training_stats['evaluation_results'][-1]['win_rate']
            improvement = final_wr - early_wr
            
            print(f"\nLearning Progress:")
            print(f"Early Win Rate: {early_wr:.1%}")
            print(f"Final Win Rate: {final_wr:.1%}")
            print(f"Improvement: {improvement:+.1%}")
            
            if improvement > 0.15:
                print("📈 Strong learning curve!")
            elif improvement > 0.05:
                print("📈 Good improvement shown")
            elif improvement > 0:
                print("📊 Modest improvement")
            else:
                print("📉 No clear improvement - may need more episodes")
        
        print(f"="*60)
    
    def _plot_quackle_training_progress(self):
        """Generate training progress plots for Quackle training"""
        
        if not self.training_stats['agent_scores']:
            print("No training data to plot")
            return
        
        plt.figure(figsize=(16, 12))
        episodes = np.arange(1, len(self.training_stats['agent_scores']) + 1)
        
        # Plot 1: Score Evolution
        plt.subplot(2, 3, 1)
        agent_scores = self.training_stats['agent_scores']
        quackle_scores = self.training_stats['quackle_scores']
        
        plt.plot(episodes, agent_scores, alpha=0.6, label='RL Agent', color='blue')
        plt.plot(episodes, quackle_scores, alpha=0.6, label='Quackle', color='orange')
        
        # Moving averages
        window = max(10, len(episodes) // 20)
        if len(episodes) >= window:
            agent_ma = moving_average(agent_scores, window)
            quackle_ma = moving_average(quackle_scores, window)
            plt.plot(episodes[window-1:], agent_ma, '--', linewidth=2, 
                    color='darkblue', label='Agent MA')
            plt.plot(episodes[window-1:], quackle_ma, '--', linewidth=2, 
                    color='darkorange', label='Quackle MA')
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Score Evolution: RL Agent vs Quackle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score Gaps
        plt.subplot(2, 3, 2)
        score_gaps = self.training_stats['score_gaps']
        plt.plot(episodes, score_gaps, alpha=0.7, color='purple')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.fill_between(episodes, score_gaps, 0, where=(np.array(score_gaps) >= 0), 
                        color='green', alpha=0.3, label='Agent Wins')
        plt.fill_between(episodes, score_gaps, 0, where=(np.array(score_gaps) < 0), 
                        color='red', alpha=0.3, label='Quackle Wins')
        
        if len(episodes) >= window:
            gap_ma = moving_average(score_gaps, window)
            plt.plot(episodes[window-1:], gap_ma, '--', linewidth=2, 
                    color='darkpurple', label='Score Gap MA')
        
        plt.xlabel('Episode')
        plt.ylabel('Score Gap (Agent - Quackle)')
        plt.title('Score Gap Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Win Rate Over Time
        plt.subplot(2, 3, 3)
        if self.training_stats['evaluation_results']:
            eval_episodes = [r['episode'] for r in self.training_stats['evaluation_results']]
            win_rates = [r['win_rate'] * 100 for r in self.training_stats['evaluation_results']]
            
            plt.plot(eval_episodes, win_rates, 'o-', linewidth=2, markersize=4, 
                    color='green', label='Win Rate vs Quackle')
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% (Even)')
            
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
            plt.title('Win Rate vs Quackle Over Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
        
        # Plot 4: TD Error Evolution
        plt.subplot(2, 3, 4)
        if self.training_stats['td_errors']:
            td_episodes = np.arange(1, len(self.training_stats['td_errors']) + 1)
            td_errors = self.training_stats['td_errors']
            
            plt.plot(td_episodes, td_errors, alpha=0.7, color='red')
            if len(td_errors) >= 10:
                td_ma = moving_average(td_errors, min(20, len(td_errors)//5))
                plt.plot(td_episodes[len(td_episodes)-len(td_ma):], td_ma, 
                        '--', linewidth=2, color='darkred', label='TD Error MA')
            
            plt.xlabel('Episode')
            plt.ylabel('TD Error')
            plt.title('TD Error Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Score Distribution Comparison
        plt.subplot(2, 3, 5)
        plt.hist(agent_scores, bins=25, alpha=0.7, label='RL Agent', 
                color='blue', density=True)
        plt.hist(quackle_scores, bins=25, alpha=0.7, label='Quackle', 
                color='orange', density=True)
        plt.axvline(np.mean(agent_scores), color='darkblue', linestyle='--', 
                   label=f'Agent Avg: {np.mean(agent_scores):.1f}')
        plt.axvline(np.mean(quackle_scores), color='darkorange', linestyle='--', 
                   label=f'Quackle Avg: {np.mean(quackle_scores):.1f}')
        
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.title('Score Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Cumulative Win Rate
        plt.subplot(2, 3, 6)
        cumulative_wins = np.cumsum(self.training_stats['wins'])
        cumulative_win_rate = cumulative_wins / episodes * 100
        
        plt.plot(episodes, cumulative_win_rate, linewidth=2, color='darkgreen')
        plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50%')
        plt.fill_between(episodes, cumulative_win_rate, 50, 
                        where=(cumulative_win_rate >= 50), 
                        color='green', alpha=0.3, label='Above 50%')
        plt.fill_between(episodes, cumulative_win_rate, 50, 
                        where=(cumulative_win_rate < 50), 
                        color='red', alpha=0.3, label='Below 50%')
        
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Win Rate (%)')
        plt.title('Cumulative Win Rate vs Quackle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f'plot/quackle_training_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Quackle training plots saved: {plot_path}")
    
    def get_training_summary(self) -> Dict:
        """Get training summary for Quackle training"""
        episodes = len(self.training_stats['agent_scores'])
        if episodes > 0:
            final_performance = {
                'win_rate': sum(self.training_stats['wins']) / episodes,
                'avg_score': np.mean(self.training_stats['agent_scores']),
                'avg_score_gap': np.mean(self.training_stats['score_gaps'])
            }
            return {
                'final_performance': final_performance,
                'training_mode': 'vs_quackle',
                'episodes_completed': episodes,
                'total_training_time': self.training_stats.get('total_training_time', 0)
            }
        
        return {'final_performance': None}
    
    def save_training_data(self, filepath: str):
        """Save Quackle training statistics"""
        training_data = {
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat(),
            'training_mode': 'vs_quackle',
            'version': '1.0_quackle_training'
        }
        save_game_data(training_data, filepath)
