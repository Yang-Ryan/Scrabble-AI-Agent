"""
Feature Extractor for Scrabble AI
Custom feature engineering designed specifically for Scrabble strategy
Converts game state + move into numerical features for machine learning
"""

import numpy as np
from typing import Dict, List, Set, Tuple

class FeatureExtractor:
    """
    Extracts strategic features from Scrabble game states and moves
    Core innovation: hand-crafted features that capture Scrabble strategy
    """
    
    def __init__(self):
        """Initialize feature extractor with feature definitions"""
        self.feature_names = [
            'Immediate Score',      # Raw points from move
            'Tile Efficiency',     # Points per tile used
            'Board Control',       # Strategic positioning value
            'Defensive Value',     # Opponent blocking potential
            'Board Openness',      # Future opportunity creation
            'Rack Balance',        # Quality of remaining tiles
            'Tile Synergy',        # Letter combination potential
            'Endgame Factor'       # Time-based strategy adjustment
        ]
        
        # Tile values for calculations
        self.tile_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, '?': 0  # Blank tile
        }
        
        # Common letter patterns for synergy calculation
        self.synergy_patterns = {
            frozenset(['Q', 'U']): 5.0,           # Q-U pair is crucial
            frozenset(['E', 'R']): 2.0,           # ER ending
            frozenset(['I', 'N', 'G']): 4.0,      # ING ending
            frozenset(['E', 'D']): 2.0,           # ED ending
            frozenset(['S', 'T', 'R']): 3.0,      # STR combination
            frozenset(['A', 'E', 'I']): 2.0,      # Multiple vowels
            frozenset(['C', 'H']): 2.5,           # CH combination
            frozenset(['T', 'H']): 2.5,           # TH combination
        }
    
    def extract_features(self, state: Dict, move: Dict) -> np.ndarray:
        """
        Extract all features for a state-action pair
        
        Args:
            state: Current game state
            move: Proposed move
            
        Returns:
            Feature vector (8-dimensional)
        """
        features = np.zeros(len(self.feature_names))
        
        try:
            # Feature 0: Immediate Score (normalized)
            features[0] = self._normalize_score(move.get('score', 0))
            
            # Feature 1: Tile Efficiency
            features[1] = self._calculate_tile_efficiency(move)
            
            # Feature 2: Board Control
            features[2] = self._calculate_board_control(state, move)
            
            # Feature 3: Defensive Value
            features[3] = self._calculate_defensive_value(state, move)
            
            # Feature 4: Board Openness
            features[4] = self._calculate_board_openness(state, move)
            
            # Feature 5: Rack Balance
            features[5] = self._calculate_rack_balance(state, move)
            
            # Feature 6: Tile Synergy
            features[6] = self._calculate_tile_synergy(state, move)
            
            # Feature 7: Endgame Factor
            features[7] = self._calculate_endgame_factor(state)
            
        except Exception as e:
            # Fallback to safe defaults if feature extraction fails
            print(f"Feature extraction error: {e}")
            features = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
        
        return features
    
    def _normalize_score(self, score: int) -> float:
        """Normalize score to [0, 1] range"""
        # Typical moves score 5-50 points, with occasional high scores
        return min(score / 50.0, 2.0)
    
    def _calculate_tile_efficiency(self, move: Dict) -> float:
        """
        Calculate points per tile used
        Higher efficiency = better tile usage
        """
        tiles_used = move.get('tiles_used', [])
        score = move.get('score', 0)
        
        if not tiles_used or len(tiles_used) == 0:
            return 0.0
        
        efficiency = score / len(tiles_used)
        return min(efficiency / 10.0, 2.0)  # Normalize and cap
    
    def _calculate_board_control(self, state: Dict, move: Dict) -> float:
        """
        Calculate strategic board positioning value
        Higher values for premium squares and central positions
        """
        control_score = 0.0
        
        # Premium squares used in this move
        premium_squares = move.get('premium_squares_used', [])
        for square_type in premium_squares:
            if square_type == 'TW':      # Triple Word
                control_score += 1.0
            elif square_type == 'DW':    # Double Word
                control_score += 0.8
            elif square_type == 'TL':    # Triple Letter
                control_score += 0.5
            elif square_type == 'DL':    # Double Letter
                control_score += 0.3
        
        # Position value (center is generally better)
        word_position = move.get('position', (7, 7))  # Default to center
        if isinstance(word_position, (list, tuple)) and len(word_position) >= 2:
            row, col = word_position[0], word_position[1]
            # Distance from center (7, 7)
            center_distance = abs(row - 7) + abs(col - 7)
            position_value = max(0, (14 - center_distance) / 14.0 * 0.5)
            control_score += position_value
        
        # Bonus for longer words (more board coverage)
        word_length = move.get('word_length', len(move.get('word', '')))
        if word_length > 4:
            control_score += (word_length - 4) * 0.1
        
        return min(control_score, 2.0)
    
    def _calculate_defensive_value(self, state: Dict, move: Dict) -> float:
        """
        Calculate how well this move blocks opponent opportunities
        """
        defensive_score = 0.0
        
        # Premium squares blocked from opponent
        premium_blocked = move.get('premium_squares_blocked', 0)
        defensive_score += premium_blocked * 0.3
        
        # High-value tiles used (removes them from future opponent access)
        high_value_tiles = {'Q', 'X', 'Z', 'J', 'K'}
        tiles_used = set(move.get('tiles_used', []))
        high_value_used = len(tiles_used.intersection(high_value_tiles))
        defensive_score += high_value_used * 0.2
        
        # Board congestion (making it harder for opponent)
        board_congestion = move.get('board_congestion_increase', 0)
        defensive_score += board_congestion * 0.1
        
        return min(defensive_score, 1.0)
    
    def _calculate_board_openness(self, state: Dict, move: Dict) -> float:
        """
        Calculate how this move affects future play opportunities
        More openness = more future options
        """
        openness_score = 0.0
        
        # Adjacent empty squares after move
        adjacent_empty = move.get('adjacent_empty_squares', 0)
        openness_score += min(adjacent_empty / 10.0, 0.8)
        
        # Word hooks created (positions where new words can be built)
        word_hooks = move.get('word_hooks_created', 0)
        openness_score += word_hooks * 0.1
        
        # Bonus for moves that don't close off large areas
        if not move.get('closes_board_section', False):
            openness_score += 0.2
        
        return min(openness_score, 1.0)
    
    def _calculate_rack_balance(self, state: Dict, move: Dict) -> float:
        """
        Evaluate quality of remaining tiles after move
        Good balance = sustainable scoring potential
        """
        remaining_rack = move.get('remaining_rack', [])
        
        if not remaining_rack:
            return 0.3  # Neutral score if no tiles remain
        
        balance_score = 0.0
        rack_size = len(remaining_rack)
        
        # Vowel-consonant balance
        vowels = sum(1 for tile in remaining_rack if tile in 'AEIOU')
        consonants = rack_size - vowels
        
        if rack_size > 0:
            vowel_ratio = vowels / rack_size
            # Optimal ratio is around 40% vowels
            balance_penalty = abs(vowel_ratio - 0.4)
            balance_score += max(0, 1.0 - balance_penalty * 2)
        
        # Penalty for too many high-value tiles (hard to use)
        high_value_tiles = sum(1 for tile in remaining_rack 
                              if tile in 'QXZJK')
        if high_value_tiles <= 1:
            balance_score += 0.3
        elif high_value_tiles >= 3:
            balance_score -= 0.2
        
        # Bonus for blank tiles (very flexible)
        blank_count = remaining_rack.count('?')
        balance_score += blank_count * 0.4
        
        # Bonus for common letters
        common_letters = sum(1 for tile in remaining_rack 
                           if tile in 'ERSTAIN')
        balance_score += min(common_letters / rack_size * 0.5, 0.3)
        
        return max(0.0, min(balance_score, 1.0))
    
    def _calculate_tile_synergy(self, state: Dict, move: Dict) -> float:
        """
        Calculate how well remaining tiles work together
        High synergy = likely to form good words together
        """
        remaining_rack = move.get('remaining_rack', [])
        
        if len(remaining_rack) <= 1:
            return 0.3  # Neutral if too few tiles
        
        synergy_score = 0.0
        rack_set = set(remaining_rack)
        
        # Check for known synergy patterns
        for pattern, value in self.synergy_patterns.items():
            if pattern.issubset(rack_set):
                synergy_score += value
        
        # Additional synergy checks
        
        # Multiple of same letter (can be good for parallel plays)
        letter_counts = {}
        for tile in remaining_rack:
            letter_counts[tile] = letter_counts.get(tile, 0) + 1
        
        duplicates = sum(1 for count in letter_counts.values() if count > 1)
        synergy_score += duplicates * 0.5
        
        # S tiles are very valuable (plurals, verb forms)
        s_count = remaining_rack.count('S')
        synergy_score += s_count * 1.0
        
        # Normalize synergy score
        return min(synergy_score / 10.0, 1.0)
    
    def _calculate_endgame_factor(self, state: Dict) -> float:
        """
        Adjust strategy based on game phase
        Early game: setup and positioning
        Mid game: balanced approach
        Endgame: maximize immediate points
        """
        tiles_remaining = state.get('tiles_remaining', 98)
        total_tiles = 98
        
        # Calculate game progress (1.0 = start, 0.0 = end)
        if total_tiles > 0:
            game_progress = tiles_remaining / total_tiles
        else:
            game_progress = 0.0
        
        # Return endgame urgency factor
        if game_progress > 0.7:
            # Early game: focus on setup (low urgency)
            return 0.2
        elif game_progress > 0.3:
            # Mid game: balanced approach
            return 0.5
        else:
            # Endgame: maximize immediate scoring (high urgency)
            return 1.0
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for analysis"""
        return self.feature_names.copy()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return detailed descriptions of each feature"""
        return {
            'Immediate Score': 'Normalized points scored by this move',
            'Tile Efficiency': 'Points per tile used (normalized)',
            'Board Control': 'Strategic value of board position and premium squares',
            'Defensive Value': 'How well move blocks opponent opportunities',
            'Board Openness': 'Future play opportunities created',
            'Rack Balance': 'Quality and balance of remaining tiles',
            'Tile Synergy': 'How well remaining tiles work together',
            'Endgame Factor': 'Game phase adjustment (early vs late game strategy)'
        }
    
    def analyze_features(self, state: Dict, move: Dict) -> Dict[str, float]:
        """
        Return detailed feature analysis for debugging/visualization
        """
        features = self.extract_features(state, move)
        feature_dict = dict(zip(self.feature_names, features))
        
        # Add raw values for debugging
        feature_dict['_raw_score'] = move.get('score', 0)
        feature_dict['_tiles_used'] = len(move.get('tiles_used', []))
        feature_dict['_remaining_rack_size'] = len(move.get('remaining_rack', []))
        feature_dict['_tiles_remaining'] = state.get('tiles_remaining', 0)
        
        return feature_dict