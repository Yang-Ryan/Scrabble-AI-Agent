"""
Training System for Scrabble RL Agent
Orchestrates the learning process through game simulation and weight updates
"""

import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from scrabble_agent import ScrabbleQLearner, GreedyAgent, RandomAgent
from move_generator import MoveGenerator
from utils import (create_empty_board, create_tile_bag, draw_tiles, 
                  place_word_on_board, get_rack_after_move, create_game_state,
                  save_game_data, format_time)

class ScrabbleTrainer:
    """
    Manages training process for Scrabble RL agent
    Plays games against baselines and updates agent weights
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        """
        Initialize trainer
        
        Args:
            dictionary_path: Path to word dictionary
        """
        self.move_generator = MoveGenerator(dictionary_path)
        
        # Training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_training_time': 0,
            'win_rates_history': [],
            'score_history': [],
            'weight_evolution': []
        }
    
    def train_agent(self, agent: ScrabbleQLearner, opponent_type: str = 'greedy',
                   num_episodes: int = 1000, evaluation_interval: int = 100,
                   save_interval: int = 500, verbose: bool = True) -> ScrabbleQLearner:
        """
        Train RL agent through self-play
        
        Args:
            agent: RL agent to train
            opponent_type: Type of opponent ('greedy', 'random', 'heuristic')
            num_episodes: Number of training episodes
            evaluation_interval: How often to evaluate progress
            save_interval: How often to save model
            verbose: Whether to print progress
            
        Returns:
            Trained agent
        """
        if verbose:
            print(f"Starting training: {num_episodes} episodes vs {opponent_type} opponent")
            print("=" * 60)
        
        start_time = time.time()
        
        # Create opponent
        opponent = self._create_opponent(opponent_type)
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Play one complete game
            game_result = self._play_training_game(agent, opponent)
            
            # Train agent on game experiences
            if game_result['experiences']:
                agent.train_on_episode(game_result['experiences'])
            
            # Update statistics
            self.training_stats['episodes_completed'] += 1
            episode_time = time.time() - episode_start_time
            
            # Periodic evaluation and reporting
            if (episode + 1) % evaluation_interval == 0:
                eval_results = self._evaluate_agent(agent, num_games=20)
                self.training_stats['win_rates_history'].append({
                    'episode': episode + 1,
                    'win_rate': eval_results['win_rate'],
                    'avg_score': eval_results['avg_score'],
                    'avg_score_gap': eval_results['avg_score_gap']
                })
                
                if verbose:
                    self._print_progress(episode + 1, eval_results, episode_time)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"rl_model_{timestamp}_ep{episode+1}.json"
                agent.save_model(model_path)
                
                if verbose:
                    print(f"Model saved: {model_path}")
        
        # Final statistics
        total_training_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_training_time
        
        if verbose:
            print("=" * 60)
            print(f"Training completed in {format_time(total_training_time)}")
            print(f"Final win rate: {self.training_stats['win_rates_history'][-1]['win_rate']:.1%}")
        
        return agent
    
    def _create_opponent(self, opponent_type: str):
        """Create opponent agent based on type"""
        if opponent_type == 'greedy':
            return GreedyAgent()
        elif opponent_type == 'random':
            return RandomAgent()
        else:
            return GreedyAgent()  # Default to greedy
    
    def _play_training_game(self, agent: ScrabbleQLearner, opponent) -> Dict:
        """
        Play one complete training game between agent and opponent
        
        Args:
            agent: RL agent
            opponent: Opponent agent
            
        Returns:
            Game result with experiences for training
        """
        # Initialize game
        board = create_empty_board()
        tile_bag = create_tile_bag()
        
        # Draw initial racks
        agent_rack = draw_tiles(tile_bag, 7)
        opponent_rack = draw_tiles(tile_bag, 7)
        
        # Game state
        agent_score = 0
        opponent_score = 0
        rounds_played = 0
        max_rounds = 50  # Prevent infinite games
        
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
                board, agent_rack, sampling_rate=0.3  # Sample for training speed
            )
            
            if valid_moves:
                # Agent chooses move
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
                    
                    # Store experience
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
                
                opponent_moves = self.move_generator.get_valid_moves(
                    board, opponent_rack, sampling_rate=0.5
                )
                
                if opponent_moves:
                    opponent_move = opponent.choose_move(
                        opponent_state, opponent_moves, training=False
                    )
                    
                    if opponent_move:
                        # Execute opponent move
                        board = place_word_on_board(
                            board, opponent_move['word'], opponent_move['positions']
                        )
                        opponent_score += opponent_move['score']
                        
                        # Update opponent rack
                        opponent_tiles_drawn = draw_tiles(tile_bag, len(opponent_move['tiles_used']))
                        opponent_rack = get_rack_after_move(
                            opponent_rack, opponent_move['tiles_used'], opponent_tiles_drawn
                        )
            
            rounds_played += 1
            
            # Early termination if no moves possible
            if not valid_moves and not opponent_moves:
                break
        
        # Mark final experience as terminal
        if agent_experiences:
            agent_experiences[-1]['terminal'] = True
        
        # Calculate final scores and winner
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
    
    def _evaluate_agent(self, agent: ScrabbleQLearner, num_games: int = 50) -> Dict:
        """
        Evaluate agent performance against multiple opponents
        
        Args:
            agent: Agent to evaluate
            num_games: Number of evaluation games
            
        Returns:
            Evaluation results
        """
        results = {
            'games_played': 0,
            'wins': 0,
            'total_score': 0,
            'total_score_gap': 0,
            'win_rate': 0.0,
            'avg_score': 0.0,
            'avg_score_gap': 0.0
        }
        
        # Test against greedy opponent
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
    
    def _play_evaluation_game(self, agent: ScrabbleQLearner, opponent) -> Dict:
        """
        Play evaluation game (no training updates)
        
        Args:
            agent: Agent to evaluate
            opponent: Opponent agent
            
        Returns:
            Game result
        """
        # Similar to training game but no experience collection
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
    
    def _print_progress(self, episode: int, eval_results: Dict, episode_time: float):
        """Print training progress"""
        print(f"Episode {episode:4d} | "
              f"Win Rate: {eval_results['win_rate']:5.1%} | "
              f"Avg Score: {eval_results['avg_score']:5.1f} | "
              f"Score Gap: {eval_results['avg_score_gap']:+5.1f} | "
              f"Time: {format_time(episode_time)}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        return {
            'training_stats': self.training_stats,
            'final_performance': self.training_stats['win_rates_history'][-1] if self.training_stats['win_rates_history'] else None
        }
    
    def save_training_data(self, filepath: str):
        """Save training statistics to file"""
        training_data = {
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        save_game_data(training_data, filepath)


def main():
    """Main training function"""
    print("Scrabble RL Agent Training")
    print("=" * 40)
    
    # Create and configure agent
    agent = ScrabbleQLearner(
        num_features=8,
        learning_rate=0.01,
        epsilon=0.3,
        gamma=0.9
    )
    
    # Create trainer
    trainer = ScrabbleTrainer('dictionary.txt')
    
    # Train agent
    trained_agent = trainer.train_agent(
        agent=agent,
        opponent_type='greedy',
        num_episodes=2000,
        evaluation_interval=100,
        save_interval=500,
        verbose=True
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"rl_model_final_{timestamp}.json"
    trained_agent.save_model(final_model_path)
    
    # Save training data
    training_data_path = f"training_data_{timestamp}.json"
    trainer.save_training_data(training_data_path)
    
    print(f"\nTraining completed!")
    print(f"Final model saved: {final_model_path}")
    print(f"Training data saved: {training_data_path}")
    
    # Print final statistics
    summary = trainer.get_training_summary()
    if summary['final_performance']:
        perf = summary['final_performance']
        print(f"\nFinal Performance:")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print(f"Average Score: {perf['avg_score']:.1f}")
        print(f"Average Score Gap: {perf['avg_score_gap']:+.1f}")


if __name__ == "__main__":
    main()