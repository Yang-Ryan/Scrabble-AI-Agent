"""
Enhanced Training System for Scrabble RL Agent
Updated to work with Experience Replay and Target Networks
"""

import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import the enhanced agent instead of the basic one
from scrabble_agent import GreedyAgent, RandomAgent
from scrabble_agent import AdaptiveScrabbleQLearner
from move_generator import MoveGenerator
from utils import (create_empty_board, create_tile_bag, draw_tiles, 
                  place_word_on_board, get_rack_after_move, create_game_state,
                  save_game_data, format_time)

class EnhancedScrabbleTrainer:
    """
    Enhanced trainer that leverages Experience Replay and Target Networks
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        self.move_generator = MoveGenerator(dictionary_path)
        
        # Enhanced training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_training_time': 0,
            'win_rates_history': [],
            'score_history': [],
            'weight_evolution': [],
            'network_analysis': [],  # Track main vs target network differences
            'buffer_statistics': [],  # Track experience buffer usage
            'td_error_evolution': []  # Track learning progress
        }
    
    def train_agent(self, agent: AdaptiveScrabbleQLearner, opponent_type: str = 'greedy',
                   num_episodes: int = 1000, evaluation_interval: int = 100,
                   save_interval: int = 500, verbose: bool = True) -> AdaptiveScrabbleQLearner:
        """
        Train enhanced RL agent with experience replay and target networks
        """
        if verbose:
            print(f"Training Enhanced Scrabble RL Agent")
            print(f"Episodes: {num_episodes} vs {opponent_type}")
            print(f"Experience Replay: Buffer size {agent.experience_buffer.max_size}, Batch size {agent.batch_size}")
            print(f"Target Network: Update every {agent.target_update_frequency} updates")
            print("=" * 70)
        
        start_time = time.time()
        opponent = self._create_opponent(opponent_type)
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Play one complete game
            game_result = self._play_training_game(agent, opponent)
            
            # Enhanced training: agent automatically handles experience replay
            if game_result['experiences']:
                agent.train_on_episode(game_result['experiences'])
            
            # Update statistics
            self.training_stats['episodes_completed'] += 1
            episode_time = time.time() - episode_start_time
            
            # Periodic evaluation and enhanced reporting
            if (episode + 1) % evaluation_interval == 0:
                eval_results = self._evaluate_agent(agent, num_games=20)
                network_analysis = agent.analyze_networks()
                
                # Store comprehensive stats
                eval_entry = {
                    'episode': episode + 1,
                    'win_rate': eval_results['win_rate'],
                    'avg_score': eval_results['avg_score'],
                    'avg_score_gap': eval_results['avg_score_gap']
                }
                self.training_stats['win_rates_history'].append(eval_entry)
                self.training_stats['network_analysis'].append({
                    'episode': episode + 1,
                    **network_analysis
                })
                
                # Track buffer usage
                buffer_stats = {
                    'episode': episode + 1,
                    'buffer_size': agent.experience_buffer.size(),
                    'buffer_utilization': agent.experience_buffer.size() / agent.experience_buffer.max_size,
                    'total_updates': agent.total_updates,
                    'target_updates': agent.target_updates
                }
                self.training_stats['buffer_statistics'].append(buffer_stats)
                
                if verbose:
                    self._print_enhanced_progress(episode + 1, eval_results, network_analysis, 
                                                buffer_stats, episode_time)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"enhanced_rl_model_{timestamp}_ep{episode+1}.json"
                agent.save_model(model_path)
                
                if verbose:
                    print(f"Enhanced model saved: {model_path}")
        
        # Final statistics
        total_training_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_training_time
        
        if verbose:
            self._print_final_summary(agent, total_training_time)
        
        return agent
    
    def _create_opponent(self, opponent_type: str):
        """Create opponent agent"""
        if opponent_type == 'greedy':
            return GreedyAgent()
        elif opponent_type == 'random':
            return RandomAgent()
        else:
            return GreedyAgent()
    
    def _play_training_game(self, agent: AdaptiveScrabbleQLearner, opponent) -> Dict:
        """
        Play training game - enhanced to properly store experiences
        """
        # Initialize game
        board = create_empty_board()
        tile_bag = create_tile_bag()
        
        agent_rack = draw_tiles(tile_bag, 7)
        opponent_rack = draw_tiles(tile_bag, 7)
        
        agent_score = 0
        opponent_score = 0
        rounds_played = 0
        max_rounds = 50
        
        # Track experiences for training
        agent_experiences = []
        
        # Game loop
        while len(tile_bag) > 0 and rounds_played < max_rounds:
            # Agent's turn
            agent_state = create_game_state(
                board, agent_rack, [], agent_score, opponent_score,
                len(tile_bag), rounds_played
            )
            
            valid_moves = self.move_generator.get_valid_moves(
                board, agent_rack, sampling_rate=0.3
            )
            
            if valid_moves:
                chosen_move = agent.choose_move(agent_state, valid_moves, training=True)
                
                if chosen_move:
                    # Execute move
                    old_state = agent_state.copy()
                    board = place_word_on_board(board, chosen_move['word'], chosen_move['positions'])
                    agent_score += chosen_move['score']
                    
                    # Update agent rack
                    tiles_drawn = draw_tiles(tile_bag, len(chosen_move['tiles_used']))
                    agent_rack = get_rack_after_move(
                        agent_rack, chosen_move['tiles_used'], tiles_drawn
                    )
                    
                    # Create new state
                    new_state = create_game_state(
                        board, agent_rack, [], agent_score, opponent_score,
                        len(tile_bag), rounds_played
                    )
                    
                    # Calculate reward
                    reward = agent.calculate_reward(old_state, new_state, chosen_move)
                    
                    # Store experience (enhanced format)
                    experience = {
                        'state': old_state,
                        'move': chosen_move,
                        'reward': reward,
                        'next_state': new_state,
                        'terminal': False
                    }
                    agent_experiences.append(experience)
            
            # Opponent's turn (unchanged)
            if len(tile_bag) > 0:
                opponent_state = create_game_state(
                    board, opponent_rack, [], opponent_score, agent_score,
                    len(tile_bag), rounds_played
                )
                
                opponent_moves = self.move_generator.get_valid_moves(
                    board, opponent_rack, sampling_rate=0.5
                )
                
                if opponent_moves:
                    opponent_move = opponent.choose_move(
                        opponent_state, opponent_moves, training=False
                    )
                    
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
        
        # Mark final experience as terminal
        if agent_experiences:
            agent_experiences[-1]['terminal'] = True
        
        # Calculate final results
        final_score_gap = agent_score - opponent_score
        agent_won = final_score_gap > 0
        
        return {
            'agent_score': agent_score,
            'opponent_score': opponent_score,
            'final_score_gap': final_score_gap,
            'agent_won': agent_won,
            'rounds_played': rounds_played,
            'experiences': agent_experiences
        }
    
    def _evaluate_agent(self, agent: AdaptiveScrabbleQLearner, num_games: int = 50) -> Dict:
        """Evaluate agent performance"""
        results = {
            'games_played': 0,
            'wins': 0,
            'total_score': 0,
            'total_score_gap': 0,
            'win_rate': 0.0,
            'avg_score': 0.0,
            'avg_score_gap': 0.0
        }
        
        greedy_opponent = GreedyAgent()
        
        for _ in range(num_games):
            game_result = self._play_evaluation_game(agent, greedy_opponent)
            
            results['games_played'] += 1
            if game_result['agent_won']:
                results['wins'] += 1
            
            results['total_score'] += game_result['agent_score']
            results['total_score_gap'] += game_result['final_score_gap']
        
        # Calculate averages
        if results['games_played'] > 0:
            results['win_rate'] = results['wins'] / results['games_played']
            results['avg_score'] = results['total_score'] / results['games_played']
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
                chosen_move = agent.choose_move(
                    agent_state, valid_moves, training=False  # No exploration
                )
                
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
    
    def _print_enhanced_progress(self, episode: int, eval_results: Dict, 
                               network_analysis: Dict, buffer_stats: Dict, episode_time: float):
        """Print enhanced training progress with network and buffer info"""
        print(f"Episode {episode:4d} | "
              f"Win Rate: {eval_results['win_rate']:5.1%} | "
              f"Avg Score: {eval_results['avg_score']:5.1f} | "
              f"Score Gap: {eval_results['avg_score_gap']:+5.1f}")
        
        print(f"             | "
              f"Buffer: {buffer_stats['buffer_size']:4d}/{buffer_stats['buffer_utilization']*100:3.0f}% | "
              f"Target Updates: {buffer_stats['target_updates']:3d} | "
              f"Net Diff: {network_analysis['weight_difference_norm']:5.3f} | "
              f"Time: {format_time(episode_time)}")
        
        # Show target network update progress
        progress = network_analysis.get('target_update_progress', 0)
        if progress > 0.8:
            print(f"             | Target network update coming soon ({progress*100:.0f}%)")
    
    def _print_final_summary(self, agent: AdaptiveScrabbleQLearner, total_time: float):
        """Print comprehensive final training summary"""
        print("=" * 70)
        print("ENHANCED TRAINING COMPLETED")
        print("=" * 70)
        print(f"Total training time: {format_time(total_time)}")
        
        # Final performance
        if self.training_stats['win_rates_history']:
            final_perf = self.training_stats['win_rates_history'][-1]
            print(f"Final win rate: {final_perf['win_rate']:.1%}")
            print(f"Final avg score: {final_perf['avg_score']:.1f}")
            print(f"Final score gap: {final_perf['avg_score_gap']:+.1f}")
        
        # Enhanced training stats
        training_stats = agent.get_training_stats()
        print(f"\nEnhanced Training Statistics:")
        print(f"  Total weight updates: {training_stats['total_updates']}")
        print(f"  Target network updates: {training_stats['target_updates']}")
        print(f"  Experience buffer size: {training_stats['buffer_size']}/{training_stats['buffer_max_size']}")
        print(f"  Main-Target weight difference: {training_stats['weight_difference']:.4f}")
        
        if 'avg_td_error' in training_stats:
            print(f"  Recent avg TD error: {training_stats['avg_td_error']:.4f}")
        
        # Network analysis
        print(f"\nNetwork Analysis:")
        final_analysis = agent.analyze_networks()
        print(f"  Network similarity: {final_analysis['network_similarity']:.3f}")
        print(f"  Max weight difference: {final_analysis['max_weight_difference']:.4f}")
        print(f"  Updates since last sync: {final_analysis['updates_since_sync']}")
        
        # Feature importance comparison
        print(f"\nFeature Importance (Main vs Target Networks):")
        main_features = agent.get_feature_importance()
        target_features = agent.get_target_feature_importance()
        
        for feature_name in main_features.keys():
            main_weight = main_features[feature_name]
            target_weight = target_features[feature_name]
            diff = abs(main_weight - target_weight)
            print(f"  {feature_name:15s}: Main {main_weight:6.3f} | Target {target_weight:6.3f} | Diff {diff:5.3f}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive enhanced training summary"""
        return {
            'training_stats': self.training_stats,
            'final_performance': self.training_stats['win_rates_history'][-1] if self.training_stats['win_rates_history'] else None,
            'final_network_analysis': self.training_stats['network_analysis'][-1] if self.training_stats['network_analysis'] else None,
            'final_buffer_stats': self.training_stats['buffer_statistics'][-1] if self.training_stats['buffer_statistics'] else None
        }
    
    def save_training_data(self, filepath: str):
        """Save enhanced training statistics"""
        training_data = {
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat(),
            'enhancement_version': '2.0_with_replay_and_target'
        }
        save_game_data(training_data, filepath)
    
    def compare_training_evolution(self) -> Dict:
        """
        Analyze how the training evolved with experience replay and target networks
        """
        if not self.training_stats['win_rates_history']:
            return {}
        
        # Performance evolution
        win_rates = [entry['win_rate'] for entry in self.training_stats['win_rates_history']]
        episodes = [entry['episode'] for entry in self.training_stats['win_rates_history']]
        
        # Network evolution
        network_diffs = []
        target_updates = []
        if self.training_stats['network_analysis']:
            network_diffs = [entry['weight_difference_norm'] for entry in self.training_stats['network_analysis']]
            target_updates = [entry['target_updates'] for entry in self.training_stats['network_analysis']]
        
        # Buffer evolution
        buffer_utilization = []
        if self.training_stats['buffer_statistics']:
            buffer_utilization = [entry['buffer_utilization'] for entry in self.training_stats['buffer_statistics']]
        
        analysis = {
            'performance_trend': {
                'initial_win_rate': win_rates[0] if win_rates else 0,
                'final_win_rate': win_rates[-1] if win_rates else 0,
                'improvement': win_rates[-1] - win_rates[0] if len(win_rates) >= 2 else 0,
                'peak_win_rate': max(win_rates) if win_rates else 0,
                'stability': np.std(win_rates[-5:]) if len(win_rates) >= 5 else float('inf')
            },
            'network_dynamics': {
                'final_weight_difference': network_diffs[-1] if network_diffs else 0,
                'max_weight_difference': max(network_diffs) if network_diffs else 0,
                'avg_weight_difference': np.mean(network_diffs) if network_diffs else 0,
                'total_target_updates': target_updates[-1] if target_updates else 0
            },
            'buffer_usage': {
                'final_buffer_utilization': buffer_utilization[-1] if buffer_utilization else 0,
                'avg_buffer_utilization': np.mean(buffer_utilization) if buffer_utilization else 0
            }
        }
        
        return analysis


def main():
    """Main enhanced training function"""
    print("Enhanced Scrabble RL Agent Training")
    print("Features: Experience Replay + Target Networks")
    print("=" * 50)
    
    # Create enhanced agent with better hyperparameters
    agent = AdaptiveScrabbleQLearner(
        num_features=8,
        learning_rate=0.01,
        epsilon=0.3,
        gamma=0.9,
        buffer_size=5000,  # Experience replay buffer
        batch_size=32,     # Batch size for replay
        target_update_frequency=100,  # Update target network every 100 updates
        min_buffer_size=500  # Start training when buffer has 500 experiences
    )
    
    # Create enhanced trainer
    trainer = EnhancedScrabbleTrainer('dictionary.txt')
    
    # Train agent with enhanced features
    trained_agent = trainer.train_agent(
        agent=agent,
        opponent_type='greedy',
        num_episodes=3000,  # More episodes to see benefit of enhancements
        evaluation_interval=100,
        save_interval=500,
        verbose=True
    )
    
    # Save final enhanced model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"enhanced_rl_model_final_{timestamp}.json"
    trained_agent.save_model(final_model_path)
    
    # Save enhanced training data
    training_data_path = f"enhanced_training_data_{timestamp}.json"
    trainer.save_training_data(training_data_path)
    
    print(f"\nEnhanced training completed!")
    print(f"Final model saved: {final_model_path}")
    print(f"Training data saved: {training_data_path}")
    
    # Print evolution analysis
    evolution_analysis = trainer.compare_training_evolution()
    if evolution_analysis:
        print(f"\nTraining Evolution Analysis:")
        perf = evolution_analysis['performance_trend']
        print(f"  Performance improvement: {perf['improvement']:.1%}")
        print(f"  Peak win rate: {perf['peak_win_rate']:.1%}")
        print(f"  Final stability (std): {perf['stability']:.3f}")
        
        network = evolution_analysis['network_dynamics']
        print(f"  Total target network updates: {network['total_target_updates']}")
        print(f"  Final main-target difference: {network['final_weight_difference']:.4f}")
        
        buffer = evolution_analysis['buffer_usage']
        print(f"  Final buffer utilization: {buffer['final_buffer_utilization']:.1%}")


if __name__ == "__main__":
    main()