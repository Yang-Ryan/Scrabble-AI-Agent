"""
Complete Adaptive Scrabble Agent
Enhanced Q-learning with Experience Replay, Target Networks, 
Adaptive Time Preferences, and Learnable Strategic Tile Values
"""

import numpy as np
import random
import json
from typing import List, Dict, Tuple, Any, Optional
from collections import deque, defaultdict

class ExperienceBuffer:
    """
    Experience Replay Buffer for storing and sampling past experiences
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_experience(self, experience: Dict):
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size

class AdaptiveTimePreference:
    """
    Learns optimal timing strategy during gameplay
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # LEARNABLE timing parameters
        self.transition_points = np.array([0.7, 0.3])  # When to switch strategies
        self.urgency_levels = np.array([0.2, 0.5, 1.0])  # How urgent each phase is
        
        # LEARNABLE context modifiers
        self.score_gap_modifier = 0.1    # How score difference affects urgency
        self.rack_quality_modifier = 0.05 # How rack quality affects timing
        self.opponent_modifier = 0.0     # Adaptation to opponent style
        
        # Track performance for learning
        self.phase_outcomes = defaultdict(list)
        self.parameter_history = []
            
    def calculate_urgency(self, game_state: Dict) -> float:

        """Calculate current urgency level based on learned parameters"""
        tiles_remaining = game_state.get('tiles_remaining', 98)
        score_gap = game_state.get('my_score', 0) - game_state.get('opponent_score', 0)
        rack_quality = self._assess_rack_quality(game_state.get('my_rack', []))
        
        # Calculate base game progress
        game_progress = 1.0 - (tiles_remaining / 98.0) if tiles_remaining <= 98 else 0.0
        
        # Determine base urgency using LEARNED transition points
        base_urgency = self._get_base_urgency(game_progress)
        
        # Apply LEARNED context modifiers
        context_adjustment = 0.0
        context_adjustment += self.score_gap_modifier * (-score_gap / 50.0)
        context_adjustment += self.rack_quality_modifier * (rack_quality - 0.5)
        context_adjustment += self.opponent_modifier
        
        # Final urgency with bounds
        final_urgency = np.clip(base_urgency + context_adjustment, 0.0, 1.0)
        return final_urgency
    
    def _get_base_urgency(self, game_progress: float) -> float:

        """Get base urgency using learned transition points and levels"""
        if game_progress >= self.transition_points[0]:
            return self.urgency_levels[0]  # Early game
        elif game_progress >= self.transition_points[1]:
            return self.urgency_levels[1]  # Mid game
        else:
            return self.urgency_levels[2]  # End game
    
    def _assess_rack_quality(self, rack: List[str]) -> float:

        """Assess rack quality (0.0 = poor, 1.0 = excellent)"""
        if not rack:
            return 0.5
        
        vowels = sum(1 for tile in rack if tile in 'AEIOU')
        high_value = sum(1 for tile in rack if tile in 'QXZJK')
        blanks = rack.count('?')
        common_letters = sum(1 for tile in rack if tile in 'ERSTAIN')
        
        vowel_ratio = vowels / len(rack)
        balanced_vowels = 1.0 if 0.2 <= vowel_ratio <= 0.5 else 0.5
        
        quality = (balanced_vowels + 
                  min(blanks * 0.3, 0.3) +
                  min(common_letters / len(rack) * 0.4, 0.4) +
                  max(0, 0.3 - high_value * 0.1))
        
        return np.clip(quality, 0.0, 1.0)
    
    def update_from_outcome(self, game_phases: List[Dict], final_outcome: Dict):
        """Learn from game outcome to improve timing strategy"""
        if not game_phases:
            return
        
        won = final_outcome.get('agent_won', False)
        score_gap = final_outcome.get('final_score_gap', 0)
        
        learning_signal = 1.0 if won else -0.5
        learning_signal += np.clip(score_gap / 100.0, -0.5, 0.5)
        
        # Update parameters based on phase effectiveness
        for phase_info in game_phases:
            phase = phase_info['phase']
            urgency_used = phase_info['urgency']
            score_gained = phase_info.get('score_gained', 0)
            
            self.phase_outcomes[phase].append({
                'urgency': urgency_used,
                'score_gained': score_gained,
                'learning_signal': learning_signal
            })
        
        self._update_timing_parameters(learning_signal, game_phases)
        
        self.parameter_history.append({
            'transition_points': self.transition_points.copy(),
            'urgency_levels': self.urgency_levels.copy(),
            'outcome': final_outcome
        })
    
    def _update_timing_parameters(self, learning_signal: float, game_phases: List[Dict]):
        """Update timing parameters based on game outcome"""
        update_size = self.learning_rate * learning_signal
        
        if learning_signal < 0:  # Poor outcome
            late_game_performance = sum(p.get('score_gained', 0) for p in game_phases 
                                      if p.get('phase') == 'late')
            
            if late_game_performance < 20:  # Poor endgame
                self.transition_points[1] = np.clip(
                    self.transition_points[1] + update_size * 0.1, 0.1, 0.5
                )
        
        # Adjust urgency levels
        for i, level in enumerate(self.urgency_levels):
            noise = np.random.normal(0, 0.02)
            adjustment = update_size * 0.1 + noise
            self.urgency_levels[i] = np.clip(level + adjustment, 0.0, 1.0)
        
        # Ensure urgency increases over time
        self.urgency_levels = np.sort(self.urgency_levels)
        
        # Update context modifiers
        self.score_gap_modifier = np.clip(
            self.score_gap_modifier + update_size * 0.05, -0.5, 0.5
        )


class AdaptiveTileValues:
    """
    Learns strategic tile values during gameplay
    """
    
    def __init__(self, learning_rate: float = 0.005):
        self.learning_rate = learning_rate
        
        # Official Scrabble point values (FIXED - these are game rules)
        self.official_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, '?': 0
        }
        
        # LEARNABLE strategic multipliers
        self.strategic_multipliers = {tile: 1.0 for tile in self.official_values.keys()}
        
        # LEARNABLE context-dependent modifiers
        self.early_game_modifiers = {tile: 0.0 for tile in self.official_values.keys()}
        self.endgame_modifiers = {tile: 0.0 for tile in self.official_values.keys()}
        
        # Track tile usage and outcomes
        self.tile_usage_outcomes = defaultdict(list)
        self.value_history = []
        
        # Initialize with strategic priors
        self._initialize_strategic_priors()
        
    
    def _initialize_strategic_priors(self):
        """Initialize with reasonable strategic priors"""
        # Vowels generally more valuable strategically
        for vowel in 'AEIOU':
            self.strategic_multipliers[vowel] = 1.2
        
        # S is extremely valuable (plurals, verb forms)
        self.strategic_multipliers['S'] = 2.0
        
        # Q is less valuable strategically (needs U)
        self.strategic_multipliers['Q'] = 0.5
        
        # Common consonants more valuable
        for letter in 'RSTLN':
            self.strategic_multipliers[letter] = 1.3
        
        # Blanks are strategically very valuable
        self.strategic_multipliers['?'] = 5.0
    
    def get_strategic_value(self, tile: str, context: Dict) -> float:
        """Get strategic value of tile in given context"""
        if tile not in self.official_values:
            return 1.0
        
        # Base strategic value
        base_value = self.official_values[tile] * self.strategic_multipliers[tile]
        
        # Context adjustments
        context_modifier = 0.0
        
        # Game phase adjustment
        game_phase = context.get('game_phase', 'mid')
        if game_phase == 'early':
            context_modifier += self.early_game_modifiers[tile]
        elif game_phase == 'late':
            context_modifier += self.endgame_modifiers[tile]
        
        # Rack synergy adjustment
        rack = context.get('rack', [])
        synergy_bonus = self._calculate_synergy_value(tile, rack)
        context_modifier += synergy_bonus
        
        # Board state adjustment
        board_openness = context.get('board_openness', 0.5)
        if board_openness > 0.7:  # Open board
            if tile in 'AEIOU' or tile == 'S':
                context_modifier += 0.2
        
        final_value = base_value + context_modifier
        return max(0.1, final_value)
    
    def _calculate_synergy_value(self, tile: str, rack: List[str]) -> float:
        """Calculate bonus value for tile based on rack synergy"""
        synergy_bonus = 0.0
        
        # Q-U synergy
        if tile == 'Q' and 'U' in rack:
            synergy_bonus += 2.0
        elif tile == 'U' and 'Q' in rack:
            synergy_bonus += 1.0
        
        # Vowel balance
        vowels_in_rack = sum(1 for t in rack if t in 'AEIOU')
        if tile in 'AEIOU':
            if vowels_in_rack < 2:
                synergy_bonus += 0.5
            elif vowels_in_rack > 4:
                synergy_bonus -= 0.3
        
        # Common ending patterns
        if tile == 'S' and len(rack) >= 4:
            synergy_bonus += 0.3
        
        if tile in 'ED' and any(t in 'DERTAILS' for t in rack):
            synergy_bonus += 0.2
        
        return synergy_bonus
    
    def update_from_tile_usage(self, tiles_used: List[str], tiles_kept: List[str], 
                             move_outcome: Dict, game_context: Dict):
        """Learn from tile usage decisions and outcomes"""
        score_gained = move_outcome.get('score', 0)
        strategic_success = move_outcome.get('strategic_success', 0.5)
        
        # Learn from tiles used
        for tile in tiles_used:
            usage_outcome = {
                'score_contribution': score_gained / len(tiles_used) if tiles_used else 0,
                'strategic_success': strategic_success,
                'context': game_context.copy()
            }
            self.tile_usage_outcomes[tile].append(usage_outcome)
        
        # Learn from tiles kept
        for tile in tiles_kept:
            keep_outcome = {
                'score_contribution': -1,
                'strategic_success': 0.3,
                'context': game_context.copy()
            }
            self.tile_usage_outcomes[tile].append(keep_outcome)
        
        # Update strategic values
        self._update_strategic_values()
    
    def _update_strategic_values(self):
        """Update strategic multipliers based on usage outcomes"""
        for tile, outcomes in self.tile_usage_outcomes.items():
            if len(outcomes) < 5:
                continue
            
            recent_outcomes = outcomes[-20:]
            avg_score = np.mean([o['score_contribution'] for o in recent_outcomes])
            avg_strategic = np.mean([o['strategic_success'] for o in recent_outcomes])
            
            performance = avg_score / 10.0 + avg_strategic
            target_multiplier = max(0.2, min(3.0, performance))
            
            current_multiplier = self.strategic_multipliers[tile]
            new_multiplier = current_multiplier + self.learning_rate * (target_multiplier - current_multiplier)
            self.strategic_multipliers[tile] = new_multiplier
            
            self._update_context_modifiers(tile, recent_outcomes)
    
    def _update_context_modifiers(self, tile: str, outcomes: List[Dict]):
        """Update context-dependent tile values"""
        early_game_performance = []
        late_game_performance = []
        
        for outcome in outcomes:
            context = outcome['context']
            performance = outcome['score_contribution'] + outcome['strategic_success']
            
            game_phase = context.get('game_phase', 'mid')
            if game_phase == 'early':
                early_game_performance.append(performance)
            elif game_phase == 'late':
                late_game_performance.append(performance)
        
        if early_game_performance:
            avg_early = np.mean(early_game_performance)
            target_modifier = (avg_early - 1.0) * 0.5
            self.early_game_modifiers[tile] += self.learning_rate * target_modifier
            self.early_game_modifiers[tile] = np.clip(self.early_game_modifiers[tile], -2.0, 2.0)
        
        if late_game_performance:
            avg_late = np.mean(late_game_performance)
            target_modifier = (avg_late - 1.0) * 0.5
            self.endgame_modifiers[tile] += self.learning_rate * target_modifier
            self.endgame_modifiers[tile] = np.clip(self.endgame_modifiers[tile], -2.0, 2.0)


class AdaptiveFeatureExtractor:
    """
    Enhanced feature extractor using adaptive components
    """
    
    def __init__(self, adaptive_timing: AdaptiveTimePreference, 
                 adaptive_tiles: AdaptiveTileValues):
        self.adaptive_timing = adaptive_timing
        self.adaptive_tiles = adaptive_tiles
        
        self.feature_names = [
            'Immediate Score',
            'Adaptive Tile Efficiency',
            'Board Control', 
            'Defensive Value',
            'Board Openness',
            'Adaptive Rack Balance',
            'Adaptive Tile Synergy',
            'Adaptive Timing Factor'
        ]
    
    def extract_features(self, state: Dict, move: Dict) -> np.ndarray:
        """Extract features with adaptive components"""
        features = np.zeros(8)
        
        # Get game context for adaptive components
        game_context = self._get_game_context(state)
        
        try:
            # Feature 0: Immediate Score (normalized)
            features[0] = min(move.get('score', 0) / 50.0, 2.0)
            
            # Feature 1: Adaptive Tile Efficiency
            features[1] = self._calculate_adaptive_tile_efficiency(move, game_context)
            
            # Feature 2: Board Control (same as before)
            features[2] = self._calculate_board_control(state, move)
            
            # Feature 3: Defensive Value (same as before)
            features[3] = self._calculate_defensive_value(state, move)
            
            # Feature 4: Board Openness (same as before)
            features[4] = self._calculate_board_openness(state, move)
            
            # Feature 5: Adaptive Rack Balance
            features[5] = self._calculate_adaptive_rack_balance(state, move, game_context)
            
            # Feature 6: Adaptive Tile Synergy
            features[6] = self._calculate_adaptive_tile_synergy(state, move, game_context)
            
            # Feature 7: Adaptive Timing Factor (REPLACES fixed endgame factor)
            features[7] = self.adaptive_timing.calculate_urgency(state)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            features = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
        
        return features
    
    def _get_game_context(self, state: Dict) -> Dict:
        """Create context for adaptive components"""
        tiles_remaining = state.get('tiles_remaining', 98)
        game_progress = 1.0 - (tiles_remaining / 98.0) if tiles_remaining <= 98 else 0.0
        
        if game_progress < 0.3:
            game_phase = 'early'
        elif game_progress < 0.7:
            game_phase = 'mid'
        else:
            game_phase = 'late'
        
        return {
            'game_phase': game_phase,
            'game_progress': game_progress,
            'tiles_remaining': tiles_remaining,
            'my_score': state.get('my_score', 0),
            'opponent_score': state.get('opponent_score', 0),
            'score_gap': state.get('my_score', 0) - state.get('opponent_score', 0),
            'rack': state.get('my_rack', []),
            'board_openness': 0.5
        }
    
    def _calculate_adaptive_tile_efficiency(self, move: Dict, game_context: Dict) -> float:
        """Enhanced tile efficiency using learned strategic tile values"""
        tiles_used = move.get('tiles_used', [])
        score = move.get('score', 0)
        
        if not tiles_used:
            return 0.0
        
        # Calculate strategic value of tiles used
        strategic_value = 0.0
        for tile in tiles_used:
            strategic_value += self.adaptive_tiles.get_strategic_value(tile, game_context)
        
        if strategic_value > 0:
            efficiency = score / strategic_value
        else:
            efficiency = score / len(tiles_used)
        
        return min(efficiency / 10.0, 2.0)
    
    def _calculate_adaptive_rack_balance(self, state: Dict, move: Dict, 
                                       game_context: Dict) -> float:
        """Enhanced rack balance using learned tile values"""
        remaining_rack = move.get('remaining_rack', [])
        
        if not remaining_rack:
            return 0.3
        
        # Calculate strategic value of remaining rack
        total_strategic_value = 0.0
        for tile in remaining_rack:
            tile_context = {**game_context, 'rack': remaining_rack}
            total_strategic_value += self.adaptive_tiles.get_strategic_value(tile, tile_context)
        
        avg_strategic_value = total_strategic_value / len(remaining_rack)
        balance_score = min(avg_strategic_value / 3.0, 1.0)
        
        return balance_score
    
    def _calculate_adaptive_tile_synergy(self, state: Dict, move: Dict, 
                                       game_context: Dict) -> float:
        """Enhanced tile synergy using learned combinations"""
        remaining_rack = move.get('remaining_rack', [])
        
        if len(remaining_rack) <= 1:
            return 0.3
        
        synergy_score = 0.0
        
        # Use adaptive tile values to assess synergy
        for tile in remaining_rack:
            tile_context = {**game_context, 'rack': remaining_rack}
            synergy_value = self.adaptive_tiles.get_strategic_value(tile, tile_context)
            synergy_score += synergy_value
        
        avg_synergy = synergy_score / len(remaining_rack)
        return min(avg_synergy / 3.0, 1.0)
    
    # Simplified versions of unchanged features
    def _calculate_board_control(self, state: Dict, move: Dict) -> float:
        """Board control calculation (unchanged)"""
        control_score = 0.0
        
        premium_squares = move.get('premium_squares_used', [])
        for square_type in premium_squares:
            if square_type == 'TW':
                control_score += 1.0
            elif square_type == 'DW':
                control_score += 0.8
            elif square_type == 'TL':
                control_score += 0.5
            elif square_type == 'DL':
                control_score += 0.3
        
        word_position = move.get('position', (7, 7))
        if isinstance(word_position, (list, tuple)) and len(word_position) >= 2:
            row, col = word_position[0], word_position[1]
            center_distance = abs(row - 7) + abs(col - 7)
            position_value = max(0, (14 - center_distance) / 14.0 * 0.5)
            control_score += position_value
        
        word_length = move.get('word_length', len(move.get('word', '')))
        if word_length > 4:
            control_score += (word_length - 4) * 0.1
        
        return min(control_score, 2.0)
    
    def _calculate_defensive_value(self, state: Dict, move: Dict) -> float:
        """Defensive value calculation (unchanged)"""
        defensive_score = 0.0
        
        premium_blocked = move.get('premium_squares_blocked', 0)
        defensive_score += premium_blocked * 0.3
        
        high_value_tiles = {'Q', 'X', 'Z', 'J', 'K'}
        tiles_used = set(move.get('tiles_used', []))
        high_value_used = len(tiles_used.intersection(high_value_tiles))
        defensive_score += high_value_used * 0.2
        
        board_congestion = move.get('board_congestion_increase', 0)
        defensive_score += board_congestion * 0.1
        
        return min(defensive_score, 1.0)
    
    def _calculate_board_openness(self, state: Dict, move: Dict) -> float:
        """Board openness calculation (unchanged)"""
        openness_score = 0.0
        
        adjacent_empty = move.get('adjacent_empty_squares', 0)
        openness_score += min(adjacent_empty / 10.0, 0.8)
        
        word_hooks = move.get('word_hooks_created', 0)
        openness_score += word_hooks * 0.1
        
        if not move.get('closes_board_section', False):
            openness_score += 0.2
        
        return min(openness_score, 1.0)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names.copy()


class AdaptiveScrabbleQLearner:
    """
    Complete Adaptive Scrabble Q-Learning Agent
    Combines Q-learning with Experience Replay, Target Networks,
    Adaptive Time Preferences, and Learnable Strategic Tile Values
    """
    
    def __init__(self, num_features: int = 8, learning_rate: float = 0.01, 
                 epsilon: float = 0.3, gamma: float = 0.9,
                 buffer_size: int = 10000, batch_size: int = 32,
                 target_update_frequency: int = 100, min_buffer_size: int = 1000):
        """Initialize adaptive agent"""
        
        # Core RL parameters
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_frequency = target_update_frequency
        self.updates_since_target_sync = 0
        self.last_td_error = 0
        
        # Networks
        self.main_weights = np.random.normal(0, 0.1, num_features)
        self.target_weights = self.main_weights.copy()
        
        # Experience replay
        self.experience_buffer = ExperienceBuffer(buffer_size)
        
        # NEW: Adaptive components
        self.adaptive_timing = AdaptiveTimePreference(learning_rate=0.01)
        self.adaptive_tiles = AdaptiveTileValues(learning_rate=0.005)
        
        # Enhanced feature extractor
        self.feature_extractor = AdaptiveFeatureExtractor(
            self.adaptive_timing, self.adaptive_tiles
        )
        
        # Training statistics
        self.training_episodes = 0
        self.total_updates = 0
        self.target_updates = 0
        
        # Track adaptive learning
        self.game_phases = []
        self.current_game_phase_info = {}
            
    def choose_move(self, state: Dict, valid_moves: List[Dict], 
                   training: bool = True) -> Optional[Dict]:
        """Choose move using adaptive Q-learning"""
        if not valid_moves:
            return None
        
        # Track decision for adaptive learning
        if training:
            self._track_game_phase(state)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Find best move using adaptive features
        best_move = None
        best_value = float('-inf')
        
        for move in valid_moves:
            # Extract features with adaptive components
            features = self.feature_extractor.extract_features(state, move)
            
            # Predict Q-value
            value = np.dot(self.main_weights, features)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else valid_moves[0]
    
    def _track_game_phase(self, state: Dict):
        """Track current game phase for adaptive learning"""
        game_context = self.feature_extractor._get_game_context(state)
        
        # Update current phase info
        self.current_game_phase_info = {
            'phase': game_context['game_phase'],
            'urgency': self.adaptive_timing.calculate_urgency(state),
            'score_before': state.get('my_score', 0),
            'tiles_remaining': state.get('tiles_remaining', 98)
        }
    
    def add_experience(self, state: Dict, move: Dict, reward: float, 
                      next_state: Optional[Dict], terminal: bool = False):
        """Add experience to replay buffer"""
        experience = {
            'state': state,
            'move': move,
            'reward': reward,
            'next_state': next_state,
            'terminal': terminal
        }
        self.experience_buffer.add_experience(experience)
    
    def calculate_reward(self, old_state: Dict, new_state: Dict, move: Dict) -> float:
        """Calculate reward with adaptive tile value considerations"""
        # Basic score gap change
        old_gap = old_state.get('my_score', 0) - old_state.get('opponent_score', 0)
        new_gap = new_state.get('my_score', 0) - new_state.get('opponent_score', 0)
        score_gap_change = new_gap - old_gap
        
        # Strategic bonuses using adaptive tile values
        strategic_bonus = 0.0
        game_context = self.feature_extractor._get_game_context(new_state)
        
        # Evaluate strategic quality of tiles used
        tiles_used = move.get('tiles_used', [])
        if tiles_used:
            strategic_cost = 0.0
            for tile in tiles_used:
                strategic_cost += self.adaptive_tiles.get_strategic_value(tile, game_context)
            
            # Bonus if we used tiles efficiently relative to their strategic value
            score_per_strategic_point = score_gap_change / strategic_cost if strategic_cost > 0 else 0
            if score_per_strategic_point > 2.0:  # Good efficiency
                strategic_bonus += 2.0
        
        # Premium square usage
        premium_squares = move.get('premium_squares_used', [])
        for square in premium_squares:
            if square == 'TW':
                strategic_bonus += 3.0
            elif square == 'DW':
                strategic_bonus += 2.0
            elif square == 'TL':
                strategic_bonus += 1.0
            elif square == 'DL':
                strategic_bonus += 0.5
        
        # Rack management with adaptive tile values
        remaining_rack = move.get('remaining_rack', [])
        if remaining_rack:
            rack_strategic_value = sum(
                self.adaptive_tiles.get_strategic_value(tile, game_context)
                for tile in remaining_rack
            )
            avg_rack_value = rack_strategic_value / len(remaining_rack)
            
            # Bonus for keeping strategically valuable tiles
            if avg_rack_value > 1.5:
                strategic_bonus += 1.0
            
            # Q without U penalty (adaptive)
            q_penalty = self.adaptive_tiles.get_strategic_value('Q', game_context)
            if 'Q' in remaining_rack and 'U' not in remaining_rack:
                strategic_bonus -= q_penalty
        
        # Word length bonus
        word_length = len(move.get('word', ''))
        if word_length >= 7:
            strategic_bonus += 5.0  # Bingo bonus
        elif word_length >= 5:
            strategic_bonus += 1.0
        
        # Timing-based adjustments using adaptive timing
        urgency = self.adaptive_timing.calculate_urgency(new_state)
        if urgency > 0.8:  # High urgency (endgame)
            # Prioritize immediate scoring
            strategic_bonus += score_gap_change * 0.2
        elif urgency < 0.3:  # Low urgency (early game)
            # Prioritize setup and rack management
            strategic_bonus += len(remaining_rack) * 0.3
        
        total_reward = score_gap_change + strategic_bonus
        return max(-50.0, min(50.0, total_reward))
    
    def train_on_batch(self):
        """Train on batch with adaptive components"""
        if not self.experience_buffer.is_ready(self.min_buffer_size):
            return
        
        batch = self.experience_buffer.sample_batch(self.batch_size)
        td_errors = []

        for experience in batch:
            state = experience['state']
            move = experience['move']
            reward = experience['reward']
            next_state = experience.get('next_state')
            terminal = experience.get('terminal', False)
            
            # Extract features using adaptive components
            features = self.feature_extractor.extract_features(state, move)
            
            # Calculate TD target using target network
            if terminal or next_state is None:
                target = reward
            else:
                next_value = self._estimate_next_state_value(next_state)
                target = reward + self.gamma * next_value
            
            # Update main network and get TD error
            td_error = self._update_main_weights(features, target)
            td_errors.append(abs(td_error))

        # Update target network periodically
        self.updates_since_target_sync += 1
        if self.updates_since_target_sync >= self.target_update_frequency:
            self.update_target_network()
            self.updates_since_target_sync = 0

        return np.mean(td_errors) if td_errors else 0.0
    
    def _estimate_next_state_value(self, next_state: Dict) -> float:
        """Estimate next state value using target network"""
        try:
            # Simplified state estimation for next state value
            score_gap = next_state.get('my_score', 0) - next_state.get('opponent_score', 0)
            tiles_remaining = next_state.get('tiles_remaining', 0)
            rack_quality = len(next_state.get('my_rack', []))
            
            # Create features for state estimation
            normalized_score_gap = max(-1.0, min(1.0, score_gap / 100.0))
            game_progress = 1.0 - (tiles_remaining / 98.0) if tiles_remaining <= 98 else 0.0
            rack_factor = rack_quality / 7.0 if rack_quality <= 7 else 1.0
            
            # Use adaptive timing for urgency
            urgency = self.adaptive_timing.calculate_urgency(next_state)
            
            state_features = np.array([
                normalized_score_gap,  # Score position
                rack_factor,          # Rack quality
                0.5,                  # Board control (simplified)
                0.3,                  # Defensive value
                0.4,                  # Board openness
                0.5,                  # Rack balance
                0.4,                  # Tile synergy
                urgency               # Adaptive timing factor
            ])
            
            return np.dot(self.target_weights, state_features)
            
        except Exception:
            return 0.0
    
    def _update_main_weights(self, features: np.ndarray, target: float) -> float:
        """Update main network weights"""
        prediction = np.dot(self.main_weights, features)
        td_error = target - prediction
        
        # Gradient descent update
        self.main_weights += self.learning_rate * td_error * features
        self.total_updates += 1
        
        return td_error
    
    def update_target_network(self):
        """Update target network"""
        self.target_weights = self.main_weights.copy()
        self.target_updates += 1
    
    def train_on_episode(self, episode_experiences: List[Dict]):
        """Process episode and train adaptive components"""
        # Add experiences to buffer
        for experience in episode_experiences:
            self.add_experience(
                experience['state'],
                experience['move'],
                experience['reward'],
                experience.get('next_state'),
                experience.get('terminal', False)
            )
        
        # Train Q-learning multiple times
        td_errors = []
        num_training_steps = min(len(episode_experiences), 5)
        
        for _ in range(num_training_steps):
            td_error = self.train_on_batch()
            if td_error is not None:  # Only append if not None
                td_errors.append(abs(td_error))
        
        # Update episode stats
        self.training_episodes += 1
        self._decay_epsilon()
        
        # Store last TD error (use 0.0 if no valid errors)
        self.last_td_error = np.mean(td_errors) if td_errors else 0.0
        
        # Process adaptive learning from episode
        self._process_adaptive_learning(episode_experiences)
    
    def get_last_td_error(self):
        return self.last_td_error
    
    
    def _process_adaptive_learning(self, episode_experiences: List[Dict]):
        """Process episode for adaptive component learning"""
        if not episode_experiences:
            return
        
        # Extract game phases and outcomes for timing learning
        game_phases = []
        tile_usage_data = []
        
        for i, exp in enumerate(episode_experiences):
            state = exp['state']
            move = exp['move']
            reward = exp['reward']
            
            # Game phase tracking
            game_context = self.feature_extractor._get_game_context(state)
            phase_info = {
                'phase': game_context['game_phase'],
                'urgency': self.adaptive_timing.calculate_urgency(state),
                'score_gained': max(0, reward),  # Positive rewards only
                'move_index': i
            }
            game_phases.append(phase_info)
            
            # Tile usage tracking
            tiles_used = move.get('tiles_used', [])
            tiles_kept = move.get('remaining_rack', [])
            move_outcome = {
                'score': move.get('score', 0),
                'strategic_success': min(1.0, max(0.0, reward / 10.0 + 0.5))  # Convert reward to 0-1 success
            }
            
            tile_usage_data.append({
                'tiles_used': tiles_used,
                'tiles_kept': tiles_kept,
                'move_outcome': move_outcome,
                'game_context': game_context
            })
        
        # Final outcome for adaptive learning
        final_experience = episode_experiences[-1]
        final_state = final_experience.get('next_state', final_experience['state'])
        final_outcome = {
            'agent_won': final_state.get('my_score', 0) > final_state.get('opponent_score', 0),
            'final_score_gap': final_state.get('my_score', 0) - final_state.get('opponent_score', 0),
            'total_score': final_state.get('my_score', 0)
        }
        
        # Update adaptive timing
        self.adaptive_timing.update_from_outcome(game_phases, final_outcome)
        
        # Update adaptive tile values
        for usage_data in tile_usage_data:
            self.adaptive_tiles.update_from_tile_usage(
                usage_data['tiles_used'],
                usage_data['tiles_kept'],
                usage_data['move_outcome'],
                usage_data['game_context']
            )
    
    def _decay_epsilon(self):
        """Decay exploration rate"""
        decay_rate = 0.995
        min_epsilon = 0.05
        
        if self.training_episodes % 50 == 0:
            self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get main network feature importance"""
        feature_names = self.feature_extractor.get_feature_names()
        return dict(zip(feature_names, self.main_weights))
    
    def get_target_feature_importance(self) -> Dict[str, float]:
        """Get target network feature importance"""
        feature_names = self.feature_extractor.get_feature_names()
        return dict(zip(feature_names, self.target_weights))
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptive learning statistics"""
        return {
            'timing_stats': self.adaptive_timing.get_timing_stats(),
            'tile_stats': self.adaptive_tiles.get_tile_stats(),
            'q_learning_stats': self.get_training_stats()
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get Q-learning training statistics"""
        return {
            'training_episodes': self.training_episodes,
            'total_updates': self.total_updates,
            'target_updates': self.target_updates,
            'current_epsilon': self.epsilon,
            'main_weights': self.main_weights.tolist(),
            'target_weights': self.target_weights.tolist(),
            'feature_importance': self.get_feature_importance(),
            'buffer_size': self.experience_buffer.size(),
            'buffer_max_size': self.experience_buffer.max_size,
            'weight_difference': np.linalg.norm(self.main_weights - self.target_weights)
        }
    
    def analyze_networks(self) -> Dict[str, Any]:
        """Analyze differences between main and target networks"""
        weight_difference = np.linalg.norm(self.main_weights - self.target_weights)
        max_weight_difference = np.max(np.abs(self.main_weights - self.target_weights))
        
        # Calculate network similarity (1.0 = identical, 0.0 = completely different)
        if np.linalg.norm(self.main_weights) > 0 and np.linalg.norm(self.target_weights) > 0:
            similarity = np.dot(self.main_weights, self.target_weights) / (
                np.linalg.norm(self.main_weights) * np.linalg.norm(self.target_weights)
            )
        else:
            similarity = 1.0
        
        return {
            'weight_difference_norm': weight_difference,
            'max_weight_difference': max_weight_difference,
            'network_similarity': abs(similarity),
            'target_updates': self.target_updates,
            'updates_since_sync': self.updates_since_target_sync,
            'target_update_progress': self.updates_since_target_sync / self.target_update_frequency
        }

    def get_timing_stats(self) -> Dict[str, Any]:
        """Get adaptive timing statistics"""
        if hasattr(self, 'adaptive_timing'):
            return {
                'transition_points': self.adaptive_timing.transition_points.tolist(),
                'urgency_levels': self.adaptive_timing.urgency_levels.tolist(),
                'score_gap_modifier': self.adaptive_timing.score_gap_modifier,
                'rack_quality_modifier': self.adaptive_timing.rack_quality_modifier,
                'parameter_updates': len(self.adaptive_timing.parameter_history)
            }
        return {}

    def get_tile_stats(self) -> Dict[str, Any]:
        """Get adaptive tile value statistics"""
        if hasattr(self, 'adaptive_tiles'):
            # Get tiles that have learned significantly different values
            changed_tiles = {}
            for tile, multiplier in self.adaptive_tiles.strategic_multipliers.items():
                if abs(multiplier - 1.0) > 0.1:  # Significantly changed from default
                    changed_tiles[tile] = multiplier
            
            return {
                'strategic_multipliers': self.adaptive_tiles.strategic_multipliers,
                'significantly_changed_tiles': changed_tiles,
                'early_game_modifiers': self.adaptive_tiles.early_game_modifiers,
                'endgame_modifiers': self.adaptive_tiles.endgame_modifiers,
                'tiles_with_learning': len(changed_tiles)
            }
        return {}
    
    def save_model(self, filepath: str):
        """Save complete adaptive model"""
        model_data = {
            # Q-learning components
            'main_weights': self.main_weights.tolist(),
            'target_weights': self.target_weights.tolist(),
            'num_features': self.num_features,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'training_episodes': self.training_episodes,
            'total_updates': self.total_updates,
            'target_updates': self.target_updates,
            
            # Adaptive timing components
            'adaptive_timing': {
                'transition_points': self.adaptive_timing.transition_points.tolist(),
                'urgency_levels': self.adaptive_timing.urgency_levels.tolist(),
                'score_gap_modifier': self.adaptive_timing.score_gap_modifier,
                'rack_quality_modifier': self.adaptive_timing.rack_quality_modifier,
                'opponent_modifier': self.adaptive_timing.opponent_modifier
            },
            
            # Adaptive tile value components
            'adaptive_tiles': {
                'strategic_multipliers': self.adaptive_tiles.strategic_multipliers,
                'early_game_modifiers': self.adaptive_tiles.early_game_modifiers,
                'endgame_modifiers': self.adaptive_tiles.endgame_modifiers
            },
            
            'model_version': 'adaptive_v1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Adaptive model saved to {filepath}")
        print(f"  Episodes trained: {self.training_episodes}")
        print(f"  Target updates: {self.target_updates}")
        print(f"  Current timing strategy: {self.adaptive_timing.urgency_levels}")
        print(f"  Strategic tile multipliers learned: {len([k for k, v in self.adaptive_tiles.strategic_multipliers.items() if abs(v - 1.0) > 0.1])}")
    
    def load_model(self, filepath: str):
        """Load complete adaptive model"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Load Q-learning components
        self.main_weights = np.array(model_data['main_weights'])
        self.target_weights = np.array(model_data.get('target_weights', model_data['main_weights']))
        self.num_features = model_data['num_features']
        self.learning_rate = model_data.get('learning_rate', 0.01)
        self.epsilon = model_data.get('epsilon', 0.1)
        self.training_episodes = model_data.get('training_episodes', 0)
        self.total_updates = model_data.get('total_updates', 0)
        self.target_updates = model_data.get('target_updates', 0)
        
        # Load adaptive timing components
        if 'adaptive_timing' in model_data:
            timing_data = model_data['adaptive_timing']
            self.adaptive_timing.transition_points = np.array(timing_data['transition_points'])
            self.adaptive_timing.urgency_levels = np.array(timing_data['urgency_levels'])
            self.adaptive_timing.score_gap_modifier = timing_data['score_gap_modifier']
            self.adaptive_timing.rack_quality_modifier = timing_data['rack_quality_modifier']
            self.adaptive_timing.opponent_modifier = timing_data['opponent_modifier']
        
        # Load adaptive tile components
        if 'adaptive_tiles' in model_data:
            tiles_data = model_data['adaptive_tiles']
            self.adaptive_tiles.strategic_multipliers = tiles_data['strategic_multipliers']
            self.adaptive_tiles.early_game_modifiers = tiles_data['early_game_modifiers']
            self.adaptive_tiles.endgame_modifiers = tiles_data['endgame_modifiers']
        
        print(f"Loaded adaptive model with {self.training_episodes} training episodes")
        print(f"  Model version: {model_data.get('model_version', 'unknown')}")
        print(f"  Adaptive timing learned: {self.adaptive_timing.urgency_levels}")
        print(f"  Adaptive tile values: {len(self.adaptive_tiles.strategic_multipliers)} tiles")


# Backward compatibility aliases
class ScrabbleQLearner(AdaptiveScrabbleQLearner):
    """Alias for backward compatibility"""
    pass

class EnhancedScrabbleQLearner(AdaptiveScrabbleQLearner):
    """Alias for backward compatibility"""
    pass

# Simple baseline agents (unchanged)
class GreedyAgent:
    """Baseline greedy agent"""
    
    def __init__(self):
        self.name = "Greedy"
    
    def choose_move(self, state: Dict, valid_moves: List[Dict], 
                   training: bool = True) -> Optional[Dict]:
        if not valid_moves:
            return None
        return max(valid_moves, key=lambda move: move.get('score', 0))

def main():
    """Test the adaptive agent"""
    print("Testing Adaptive Scrabble Q-Learning Agent")
    print("=" * 50)
    
    # Create adaptive agent
    agent = AdaptiveScrabbleQLearner(
        num_features=8,
        learning_rate=0.01,
        epsilon=0.3,
        gamma=0.9,
        buffer_size=5000,
        batch_size=32,
        target_update_frequency=100
    )
    
    # Test state and moves
    test_state = {
        'my_score': 50,
        'opponent_score': 45,
        'tiles_remaining': 60,
        'my_rack': ['C', 'A', 'R', 'E', 'S', 'T', 'N']
    }
    
    test_moves = [
        {
            'word': 'CARE',
            'score': 6,
            'tiles_used': ['C', 'A', 'R', 'E'],
            'remaining_rack': ['S', 'T', 'N'],
            'premium_squares_used': [],
            'position': (7, 7)
        },
        {
            'word': 'CARES',
            'score': 8,
            'tiles_used': ['C', 'A', 'R', 'E', 'S'],
            'remaining_rack': ['T', 'N'],
            'premium_squares_used': ['DW'],
            'position': (7, 6)
        }
    ]
    
    # Test move selection
    print("Testing move selection...")
    chosen_move = agent.choose_move(test_state, test_moves, training=False)
    if chosen_move:
        print(f"Chosen move: {chosen_move['word']} (score: {chosen_move['score']})")
    
    # Test feature extraction
    print("\nTesting adaptive feature extraction...")
    features = agent.feature_extractor.extract_features(test_state, test_moves[0])
    feature_names = agent.feature_extractor.get_feature_names()
    
    print("Features extracted:")
    for name, value in zip(feature_names, features):
        print(f"  {name}: {value:.3f}")
    
    # Test adaptive components
    print(f"\nAdaptive timing urgency: {agent.adaptive_timing.calculate_urgency(test_state):.3f}")
    
    print("\nSample strategic tile values:")
    game_context = agent.feature_extractor._get_game_context(test_state)
    for tile in ['S', 'Q', 'E', 'A']:
        value = agent.adaptive_tiles.get_strategic_value(tile, game_context)
        print(f"  {tile}: {value:.2f}")
    
    print("\nAdaptive agent initialized successfully!")


if __name__ == "__main__":
    main()