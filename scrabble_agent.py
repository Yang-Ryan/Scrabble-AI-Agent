"""
Scrabble AI Agent - Core reinforcement learning implementation
Custom Q-learning with linear function approximation
Optimizes for score gap maximization
"""

import numpy as np
import random
import json
from typing import List, Dict, Tuple, Any, Optional
from feature_extractor import FeatureExtractor

class ScrabbleQLearner:
    """
    Custom Q-Learning agent using linear function approximation
    Implements gradient descent from scratch (no pre-built ML libraries)
    """
    
    def __init__(self, num_features: int = 8, learning_rate: float = 0.01, 
                 epsilon: float = 0.3, gamma: float = 0.9):
        """
        Initialize the Q-Learning agent
        
        Args:
            num_features: Number of features in feature vector
            learning_rate: Step size for gradient descent
            epsilon: Exploration rate for epsilon-greedy
            gamma: Discount factor for future rewards
        """
        # Core RL parameters
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.gamma = gamma
        
        # Linear function approximation weights
        self.weights = np.random.normal(0, 0.1, num_features)
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor()
        
        # Training statistics
        self.training_episodes = 0
        self.total_updates = 0
        self.weight_history = []
        self.performance_history = []
        
        # Experience replay buffer (optional)
        self.experience_buffer = []
        self.buffer_size = 1000
    
    def predict_value(self, features: np.ndarray) -> float:
        """
        Predict Q-value using linear function approximation
        Q(s,a) = w^T * φ(s,a)
        
        Args:
            features: Feature vector φ(s,a)
            
        Returns:
            Predicted Q-value (expected future score gap)
        """
        return np.dot(self.weights, features)
    
    def choose_move(self, state: Dict, valid_moves: List[Dict], 
                   training: bool = True) -> Optional[Dict]:
        """
        Choose best move using epsilon-greedy policy
        
        Args:
            state: Current game state
            valid_moves: List of valid moves
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected move or None if no valid moves
        """
        if not valid_moves:
            return None
        
        # Epsilon-greedy exploration (only during training)
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Exploit: choose move with highest predicted value
        best_move = None
        best_value = float('-inf')
        
        for move in valid_moves:
            features = self.feature_extractor.extract_features(state, move)
            value = self.predict_value(features)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else valid_moves[0]
    
    def calculate_reward(self, old_state: Dict, new_state: Dict, 
                        move: Dict) -> float:
        """
        Calculate reward based on score gap change (CORE INNOVATION)
        
        Reward = Δ(my_score - opponent_score) + strategic bonuses
        
        Args:
            old_state: State before move
            new_state: State after move
            move: Move that was executed
            
        Returns:
            Reward signal for learning
        """
        # Primary reward: change in score gap
        old_gap = old_state.get('my_score', 0) - old_state.get('opponent_score', 0)
        new_gap = new_state.get('my_score', 0) - new_state.get('opponent_score', 0)
        score_gap_change = new_gap - old_gap
        
        # Strategic bonuses (small rewards for good positioning)
        strategic_bonus = 0
        
        # Bonus for using premium squares
        if move.get('premium_squares_used', []):
            strategic_bonus += len(move['premium_squares_used']) * 0.5
        
        # Bonus for maintaining rack balance
        remaining_rack = move.get('remaining_rack', [])
        if remaining_rack:
            vowels = sum(1 for tile in remaining_rack if tile in 'AEIOU')
            if len(remaining_rack) > 0:
                vowel_ratio = vowels / len(remaining_rack)
                if 0.3 <= vowel_ratio <= 0.5:  # Good balance
                    strategic_bonus += 1.0
        
        # Small penalty for ending with difficult tiles
        penalty = 0
        if 'Q' in remaining_rack and 'U' not in remaining_rack:
            penalty -= 2.0
        
        return score_gap_change + strategic_bonus + penalty
    
    def update_weights(self, features: np.ndarray, target: float):
        """
        Update weights using gradient descent
        
        Implements: w ← w + α * (target - prediction) * φ(s,a)
        
        Args:
            features: Feature vector φ(s,a)
            target: Target Q-value
        """
        # Current prediction
        prediction = self.predict_value(features)
        
        # Compute TD error
        td_error = target - prediction
        
        # Gradient descent update
        self.weights += self.learning_rate * td_error * features
        
        # Update statistics
        self.total_updates += 1
        
        # Store weight history periodically for analysis
        if self.total_updates % 500 == 0:
            self.weight_history.append({
                'episode': self.training_episodes,
                'weights': self.weights.copy(),
                'td_error': abs(td_error)
            })
    
    def train_on_episode(self, episode_experiences: List[Dict]):
        """
        Train on a complete episode using temporal difference learning
        
        Args:
            episode_experiences: List of experience dictionaries
        """
        # Process experiences in reverse order for TD updates
        final_score_gap = 0
        if episode_experiences:
            final_exp = episode_experiences[-1]
            final_state = final_exp['next_state']
            if final_state:
                final_score_gap = (final_state.get('my_score', 0) - 
                                 final_state.get('opponent_score', 0))
        
        # Update each experience
        for i, experience in enumerate(episode_experiences):
            state = experience['state']
            move = experience['move']
            reward = experience['reward']
            next_state = experience.get('next_state')
            terminal = experience.get('terminal', False)
            
            # Extract features
            features = self.feature_extractor.extract_features(state, move)
            
            # Calculate TD target
            if terminal or next_state is None:
                # Terminal state: target = reward
                target = reward
            else:
                # Non-terminal: target = reward + γ * estimated_future_value
                # For simplicity, use final score gap as future value estimate
                remaining_episodes = len(episode_experiences) - i - 1
                future_discount = self.gamma ** max(1, remaining_episodes / 10.0)
                target = reward + future_discount * final_score_gap / 10.0
            
            # Update weights
            self.update_weights(features, target)
        
        # Update episode statistics
        self.training_episodes += 1
        
        # Decay exploration rate
        self._decay_epsilon()
    
    def _decay_epsilon(self):
        """Decay exploration rate over time"""
        # Exponential decay
        decay_rate = 0.995
        min_epsilon = 0.05
        
        if self.training_episodes % 50 == 0:
            self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def add_experience(self, experience: Dict):
        """Add experience to replay buffer"""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def replay_experiences(self, batch_size: int = 32):
        """Sample and replay experiences from buffer"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        for experience in batch:
            state = experience['state']
            move = experience['move']
            reward = experience['reward']
            
            features = self.feature_extractor.extract_features(state, move)
            target = reward  # Simplified target
            
            self.update_weights(features, target)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return current feature weights for analysis"""
        feature_names = self.feature_extractor.get_feature_names()
        return dict(zip(feature_names, self.weights))
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Return training statistics for analysis"""
        return {
            'training_episodes': self.training_episodes,
            'total_updates': self.total_updates,
            'current_epsilon': self.epsilon,
            'current_weights': self.weights.tolist(),
            'feature_importance': self.get_feature_importance()
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'weights': self.weights.tolist(),
            'num_features': self.num_features,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'gamma': self.gamma,
            'training_episodes': self.training_episodes,
            'total_updates': self.total_updates,
            'feature_names': self.feature_extractor.get_feature_names()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.weights = np.array(model_data['weights'])
        self.num_features = model_data['num_features']
        self.learning_rate = model_data.get('learning_rate', 0.01)
        self.epsilon = model_data.get('epsilon', 0.1)
        self.training_episodes = model_data.get('training_episodes', 0)
        self.total_updates = model_data.get('total_updates', 0)
        
        print(f"Loaded model with {self.training_episodes} training episodes")


class GreedyAgent:
    """
    Baseline greedy agent that always chooses highest-scoring move
    Used as training opponent and evaluation baseline
    """
    
    def __init__(self):
        self.name = "Greedy"
    
    def choose_move(self, state: Dict, valid_moves: List[Dict], 
                   training: bool = True) -> Optional[Dict]:
        """Always choose move with highest immediate score"""
        if not valid_moves:
            return None
        
        # Find move with maximum score
        best_move = max(valid_moves, key=lambda move: move.get('score', 0))
        return best_move


class RandomAgent:
    """
    Random baseline agent for testing and lower bound comparison
    """
    
    def __init__(self):
        self.name = "Random"
    
    def choose_move(self, state: Dict, valid_moves: List[Dict], 
                   training: bool = True) -> Optional[Dict]:
        """Choose random valid move"""
        if not valid_moves:
            return None
        return random.choice(valid_moves)


class HeuristicAgent:
    """
    More sophisticated baseline using simple heuristics
    Considers both score and rack balance
    """
    
    def __init__(self):
        self.name = "Heuristic"
        self.feature_extractor = FeatureExtractor()
    
    def choose_move(self, state: Dict, valid_moves: List[Dict], 
                   training: bool = True) -> Optional[Dict]:
        """Choose move based on simple heuristic scoring"""
        if not valid_moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            # Simple heuristic: immediate score + rack balance
            immediate_score = move.get('score', 0)
            
            # Bonus for good rack balance
            features = self.feature_extractor.extract_features(state, move)
            rack_balance = features[5] if len(features) > 5 else 0  # Rack balance feature
            
            # Combined score
            heuristic_score = immediate_score + rack_balance * 10
            
            if heuristic_score > best_score:
                best_score = heuristic_score
                best_move = move
        
        return best_move