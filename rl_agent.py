import numpy as np
import random
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import math

from scrabble_game import Player, ScrabbleGame
from opponent_modeling import OpponentModel

class QLearningAgent(Player):
    """Q-Learning based Scrabble agent with opponent modeling"""
    
    def __init__(self, name: str, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        super().__init__(name)
        
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-table and experience
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_replay = deque(maxlen=10000)
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        
        # Opponent modeling
        self.opponent_model = None
        self.game_history = []
        
        # Strategy parameters
        self.offensive_weight = 0.7
        self.defensive_weight = 0.3
        self.exploration_bonus = 0.1
        
        # Performance tracking
        self.games_played = 0
        self.total_reward = 0
        self.move_history = []
    
    def initialize_opponent_model(self, initial_tile_distribution: Dict[str, int]):
        """Initialize opponent modeling system"""
        self.opponent_model = OpponentModel(initial_tile_distribution)
    
    def get_state_representation(self, game: ScrabbleGame) -> str:
        """Convert game state to string representation for Q-table"""
        state_features = []
        
        # Board occupancy features (simplified)
        board = game.board.board
        occupied_positions = 0
        premium_occupied = 0
        center_occupied = 1 if board[7][7] != '' else 0
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != '':
                    occupied_positions += 1
                    if (i, j) in game.board.premium_squares:
                        premium_occupied += 1
        
        state_features.extend([
            occupied_positions // 5,  # Discretize board occupancy
            premium_occupied,
            center_occupied
        ])
        
        # Player tile features
        tile_counts = defaultdict(int)
        for tile in self.tiles:
            tile_counts[tile] += 1
        
        # High-value tiles
        high_value_tiles = sum(tile_counts[letter] for letter in ['J', 'Q', 'X', 'Z'])
        vowel_count = sum(tile_counts[letter] for letter in ['A', 'E', 'I', 'O', 'U'])
        
        state_features.extend([
            len(self.tiles),
            high_value_tiles,
            vowel_count,
            min(tile_counts.get('_', 0), 2)  # Blank tiles (capped at 2)
        ])
        
        # Game progress features
        tiles_remaining = game.tile_bag.remaining_count()
        turn_phase = 0 if tiles_remaining > 70 else (1 if tiles_remaining > 30 else 2)
        
        state_features.extend([
            turn_phase,
            game.turn_count // 10  # Discretize turn count
        ])
        
        # Score difference
        if len(game.players) > 1:
            opponent_score = max(p.score for p in game.players if p != self)
            score_diff = (self.score - opponent_score) // 10  # Discretize score difference
            state_features.append(max(-10, min(10, score_diff)))
        else:
            state_features.append(0)
        
        return '_'.join(map(str, state_features))
    
    def get_action_representation(self, move: Dict) -> str:
        """Convert move to string representation"""
        if not move:
            return "PASS"
        
        word = move['word']
        direction = move['direction']
        score_range = move['score'] // 5  # Discretize score into ranges
        
        # Categorize move type
        move_type = self._categorize_move(move)
        
        return f"{len(word)}_{direction}_{score_range}_{move_type}"
    
    def _categorize_move(self, move: Dict) -> str:
        """Categorize move by strategic type"""
        word = move['word']
        score = move['score']
        
        # High scoring move
        if score >= 20:
            return "HIGH_SCORE"
        
        # Defensive move (low score but strategic)
        if score < 10 and len(word) >= 4:
            return "DEFENSIVE"
        
        # Vowel dump
        vowels = sum(1 for letter in word if letter in 'AEIOU')
        if vowels >= len(word) * 0.6:
            return "VOWEL_DUMP"
        
        # Consonant heavy
        if vowels <= len(word) * 0.3:
            return "CONSONANT_HEAVY"
        
        return "STANDARD"
    
    def calculate_reward(self, game: ScrabbleGame, move: Dict, prev_state: Dict) -> float:
        """Calculate reward for a move with normalization and penalty for bad moves"""
        if not move:
            return -5  # Penalty for passing

        # ✔️ 基礎得分 = move 實際分數
        base_reward = move['score']

        # ✔️ 額外小獎勵：Bingo / 難字
        if len(move['word']) >= 7:
            base_reward += 5  # small bingo bonus

        high_value_letters = {'J', 'Q', 'X', 'Z'}
        for letter in move['word']:
            if letter in high_value_letters:
                base_reward += 1  # bonus for rare letters

        # ✔️ Tile balance penalty
        vowel_count = sum(1 for tile in self.tiles if tile in 'AEIOU')
        consonant_count = len(self.tiles) - vowel_count
        if vowel_count >= 6 or consonant_count >= 6:
            base_reward -= 2

        # ✔️ Normalization：不要讓 reward 太誇張
        base_reward = max(-10, min(base_reward, 30))

        return base_reward

    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get agent's move using Q-learning with opponent modeling"""
        valid_moves = game.get_valid_moves(self.tiles)
        
        if not valid_moves:
            return None  # Pass turn
        
        current_state = self.get_state_representation(game)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random move with strategic bias
            move = self._explore_with_bias(valid_moves, game)
        else:
            # Exploitation: best Q-value move
            move = self._select_best_move(valid_moves, current_state, game)
        
        # Store state-action for learning
        if move:
            action = self.get_action_representation(move)
            self.last_state = current_state
            self.last_action = action
            self.last_move = move
        
        return move
    
    def _explore_with_bias(self, valid_moves: List[Dict], game: ScrabbleGame) -> Dict:
        """Smart exploration with strategic bias"""
        if not valid_moves:
            return None
        
        # Weight moves by potential
        move_weights = []
        for move in valid_moves:
            weight = move['score']  # Base weight on score
            
            # Add strategic bonuses
            if self.opponent_model:
                # Defensive bonus
                defensive_positions = self.opponent_model.get_defensive_positions(
                    game.board.board,
                    self.opponent_model.predict_opponent_tiles()
                )
                if move['position'] in defensive_positions:
                    weight += 15
            
            # Exploration bonus for less tried moves
            state = self.get_state_representation(game)
            action = self.get_action_representation(move)
            count = self.state_action_counts[state][action]
            exploration_bonus = self.exploration_bonus / (1 + count)
            weight += exploration_bonus * 10
            
            move_weights.append(max(1, weight))
        
        # Weighted random selection
        total_weight = sum(move_weights)
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(move_weights):
            cumulative += weight
            if r <= cumulative:
                return valid_moves[i]
        
        return valid_moves[-1]
    
    def _select_best_move(self, valid_moves: List[Dict], state: str, game: ScrabbleGame) -> Dict:
        """Select move with highest Q-value"""
        best_move = None
        best_q_value = float('-inf')
        
        for move in valid_moves:
            action = self.get_action_representation(move)
            q_value = self.q_table[state][action]
            
            # Add opponent modeling considerations
            if self.opponent_model:
                opponent_tiles = self.opponent_model.predict_opponent_tiles()
                
                # Defensive adjustment
                defensive_positions = self.opponent_model.get_defensive_positions(
                    game.board.board, opponent_tiles
                )
                if move['position'] in defensive_positions:
                    q_value += 5  # Defensive bonus
                
                # Offensive adjustment based on opponent's likely response
                opponent_response_quality = self._estimate_opponent_response_quality(
                    move, game, opponent_tiles
                )
                q_value -= opponent_response_quality * 0.3  # Reduce value if opponent can respond well
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = move
        
        return best_move
    
    def _estimate_opponent_response_quality(self, our_move: Dict, game: ScrabbleGame, 
                                         opponent_tiles: Dict[str, float]) -> float:
        """Estimate quality of opponent's likely response to our move"""
        # Simulate board after our move
        temp_board = [row[:] for row in game.board.board]
        
        # This is a simplified estimation
        # In practice, you'd simulate the move and check opponent's options
        
        # Base estimation on move score and position
        move_score = our_move['score']
        position = our_move['position']
        
        # If we play near premium squares, opponent might benefit
        premium_squares = game.board.premium_squares
        nearby_premium = 0
        
        row, col = position
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_pos = (row + dr, col + dc)
                if new_pos in premium_squares:
                    nearby_premium += 1
        
        # Estimate opponent response quality
        response_quality = move_score * 0.1 + nearby_premium * 2
        
        return min(response_quality, 10)  # Cap the penalty
    
    def update_q_value(self, reward: float, next_state: str, game_over: bool = False):
        """Update Q-value using Q-learning formula"""
        if not hasattr(self, 'last_state') or not hasattr(self, 'last_action'):
            return
        
        current_q = self.q_table[self.last_state][self.last_action]
        
        if game_over:
            next_max_q = 0
        else:
            # Get maximum Q-value for next state
            next_actions = self.q_table[next_state]
            next_max_q = max(next_actions.values()) if next_actions else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[self.last_state][self.last_action] = new_q
        
        # Update state-action counts
        self.state_action_counts[self.last_state][self.last_action] += 1
        
        # Store experience for replay
        if hasattr(self, 'last_move'):
            experience = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': next_state,
                'done': game_over,
                'move': self.last_move
            }
            self.experience_replay.append(experience)
    
    def experience_replay_update(self, batch_size: int = 32):
        """Update Q-values using experience replay"""
        if len(self.experience_replay) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.experience_replay, batch_size)
        
        for experience in batch:
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state = experience['next_state']
            done = experience['done']
            
            current_q = self.q_table[state][action]
            
            if done:
                target_q = reward
            else:
                next_actions = self.q_table[next_state]
                max_next_q = max(next_actions.values()) if next_actions else 0
                target_q = reward + self.discount_factor * max_next_q
            
            # Update Q-value
            self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def update_opponent_model(self, opponent_move: Dict, board_before, board_after):
        """Update opponent model based on observed move"""
        if self.opponent_model and opponent_move:
            self.opponent_model.observe_move(
                opponent_move['word'],
                opponent_move['position'],
                opponent_move['direction'],
                board_before,
                board_after
            )
    
    def end_game_update(self, final_reward: float, won: bool):
        """Final update at end of game"""
        self.games_played += 1
        self.total_reward += final_reward
        
        # Large reward/penalty for winning/losing
        game_outcome_reward = 50 if won else -20
        final_reward += game_outcome_reward
        
        # Final Q-value update
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            current_q = self.q_table[self.last_state][self.last_action]
            self.q_table[self.last_state][self.last_action] = current_q + self.learning_rate * (
                final_reward - current_q
            )
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Experience replay update
        self.experience_replay_update()
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'games_played': self.games_played,
            'total_reward': self.total_reward,
            'state_action_counts': dict(self.state_action_counts)
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Convert back to defaultdict
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in model_data['q_table'].items():
                for action, q_value in actions.items():
                    self.q_table[state][action] = q_value
            
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.games_played = model_data.get('games_played', 0)
            self.total_reward = model_data.get('total_reward', 0)
            
            # Load state-action counts
            self.state_action_counts = defaultdict(lambda: defaultdict(int))
            if 'state_action_counts' in model_data:
                for state, actions in model_data['state_action_counts'].items():
                    for action, count in actions.items():
                        self.state_action_counts[state][action] = count
            
            print(f"Model loaded: {self.games_played} games played, epsilon={self.epsilon:.3f}")
            
        except FileNotFoundError:
            print(f"Model file {filepath} not found, starting fresh")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_reward = self.total_reward / max(1, self.games_played)
        
        return {
            'games_played': self.games_played,
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'total_state_actions': sum(
                len(actions) for actions in self.q_table.values()
            )
        }