import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import math

class OpponentModel:
    """Models opponent's likely tiles based on observed moves"""
    
    def __init__(self, initial_tile_distribution: Dict[str, int]):
        self.initial_distribution = initial_tile_distribution.copy()
        self.observed_tiles = Counter()  # Tiles seen on board
        self.opponent_possible_tiles = self.initial_distribution.copy()
        self.move_history = []  # Track opponent moves
        self.tile_probabilities = {}
        self.update_probabilities()
    
    def update_probabilities(self):
        """Update probability distribution of opponent's tiles"""
        total_remaining = sum(self.opponent_possible_tiles.values())
        if total_remaining == 0:
            self.tile_probabilities = {}
            return
            
        self.tile_probabilities = {
            letter: count / total_remaining 
            for letter, count in self.opponent_possible_tiles.items()
            if count > 0
        }
    
    def observe_move(self, word: str, position: Tuple[int, int], direction: str, board_before, board_after):
        """Update model based on observed opponent move"""
        # Extract tiles actually placed by opponent
        placed_tiles = self._extract_placed_tiles(word, position, direction, board_before, board_after)
        
        # Update observed tiles
        for tile in placed_tiles:
            self.observed_tiles[tile] += 1
            if tile in self.opponent_possible_tiles:
                self.opponent_possible_tiles[tile] = max(0, self.opponent_possible_tiles[tile] - 1)
        
        # Record move in history
        self.move_history.append({
            'word': word,
            'tiles_used': placed_tiles,
            'position': position,
            'direction': direction,
            'turn': len(self.move_history)
        })
        
        self.update_probabilities()
    
    def _extract_placed_tiles(self, word: str, position: Tuple[int, int], direction: str, 
                             board_before, board_after) -> List[str]:
        """Extract which tiles were actually placed (not already on board)"""
        placed_tiles = []
        row, col = position
        
        for i, letter in enumerate(word):
            if direction == 'horizontal':
                tile_pos = (row, col + i)
            else:
                tile_pos = (row + i, col)
            
            # If this position was empty before and filled after, it was placed
            if (board_before[tile_pos[0]][tile_pos[1]] == '' and 
                board_after[tile_pos[0]][tile_pos[1]] == letter):
                placed_tiles.append(letter)
        
        return placed_tiles
    
    def predict_opponent_tiles(self, num_tiles: int = 7) -> Dict[str, float]:
        """Predict probability distribution of opponent's current tiles"""
        if not self.tile_probabilities:
            return {}
        
        # Use multinomial distribution to predict most likely tile combinations
        predicted_distribution = {}
        
        # Simple approximation: assume each tile slot is independent
        for letter, prob in self.tile_probabilities.items():
            # Expected count of this letter in opponent's hand
            expected_count = prob * num_tiles
            predicted_distribution[letter] = expected_count
        
        return predicted_distribution
    
    def estimate_opponent_move_value(self, possible_moves: List[Dict], 
                                   opponent_tiles_prediction: Dict[str, float]) -> Dict[int, float]:
        """Estimate likelihood of opponent making each possible move"""
        move_likelihoods = {}
        
        for i, move in enumerate(possible_moves):
            word = move['word']
            score = move['score']
            
            # Calculate probability opponent can make this move
            can_make_prob = self._calculate_move_probability(word, opponent_tiles_prediction)
            
            # Weight by expected utility (score)
            utility_weight = self._score_to_utility(score)
            
            # Combine probability and utility
            move_likelihoods[i] = can_make_prob * utility_weight
        
        return move_likelihoods
    
    def _calculate_move_probability(self, word: str, tile_prediction: Dict[str, float]) -> float:
        """Calculate probability opponent can make a specific word"""
        word_tiles = Counter(word)
        probability = 1.0
        
        for letter, count_needed in word_tiles.items():
            if letter in tile_prediction:
                # Approximate probability using Poisson distribution
                expected_count = tile_prediction[letter]
                if expected_count > 0:
                    # P(having at least count_needed of this letter)
                    prob = min(1.0, expected_count / count_needed)
                    probability *= prob
                else:
                    probability = 0.0
                    break
            else:
                probability = 0.0
                break
        
        return probability
    
    def _score_to_utility(self, score: int) -> float:
        """Convert score to utility weight"""
        # Use sigmoid function to normalize scores
        return 1.0 / (1.0 + math.exp(-score / 10.0))
    
    def get_defensive_positions(self, board, opponent_tiles_prediction: Dict[str, float]) -> List[Tuple[int, int]]:
        """Identify positions to block based on opponent model"""
        defensive_positions = []
        
        # Find positions that could lead to high-scoring opponent moves
        premium_squares = [(0,0), (0,7), (0,14), (7,0), (7,14), (14,0), (14,7), (14,14)]  # TWS
        premium_squares.extend([(1,1), (2,2), (3,3), (4,4)])  # Some DWS
        
        for pos in premium_squares:
            row, col = pos
            if board[row][col] == '':  # Position is empty
                # Check if opponent likely has tiles to use this position
                adjacent_letters = self._get_adjacent_letters(board, pos)
                if adjacent_letters:
                    # Calculate threat level
                    threat_level = self._calculate_position_threat(pos, adjacent_letters, opponent_tiles_prediction)
                    if threat_level > 0.5:  # Threshold for defensive action
                        defensive_positions.append(pos)
        
        return defensive_positions
    
    def _get_adjacent_letters(self, board, position: Tuple[int, int]) -> List[str]:
        """Get letters adjacent to a position"""
        row, col = position
        adjacent = []
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                if board[nr][nc] != '':
                    adjacent.append(board[nr][nc])
        
        return adjacent
    
    def _calculate_position_threat(self, position: Tuple[int, int], 
                                 adjacent_letters: List[str], 
                                 opponent_tiles: Dict[str, float]) -> float:
        """Calculate threat level of a position"""
        # Simplified threat calculation
        threat = 0.0
        
        # Higher threat if opponent likely has complementary tiles
        for letter in adjacent_letters:
            # Check common letter combinations
            common_pairs = {'Q': 'U', 'S': 'T', 'E': 'R'}
            if letter in common_pairs:
                complement = common_pairs[letter]
                if complement in opponent_tiles:
                    threat += opponent_tiles[complement]
        
        # Add base threat for premium positions
        premium_positions = [(0,0), (0,7), (0,14), (7,0), (7,14), (14,0), (14,7), (14,14)]
        if position in premium_positions:
            threat += 0.3
        
        return min(threat, 1.0)
    
    def get_strategic_insights(self) -> Dict[str, any]:
        """Get strategic insights about opponent"""
        insights = {
            'playing_style': self._analyze_playing_style(),
            'preferred_word_lengths': self._analyze_word_preferences(),
            'high_value_letter_usage': self._analyze_letter_usage(),
            'defensive_tendency': self._analyze_defensive_play()
        }
        
        return insights
    
    def _analyze_playing_style(self) -> str:
        """Analyze opponent's playing style based on move history"""
        if len(self.move_history) < 3:
            return "insufficient_data"
        
        avg_score = np.mean([self._estimate_move_score(move) for move in self.move_history])
        
        if avg_score > 25:
            return "aggressive"
        elif avg_score > 15:
            return "balanced"
        else:
            return "conservative"
    
    def _analyze_word_preferences(self) -> Dict[int, int]:
        """Analyze preferred word lengths"""
        length_counts = Counter()
        for move in self.move_history:
            length_counts[len(move['word'])] += 1
        return dict(length_counts)
    
    def _analyze_letter_usage(self) -> Dict[str, int]:
        """Analyze usage of high-value letters"""
        high_value_letters = {'J', 'Q', 'X', 'Z'}
        usage = Counter()
        
        for move in self.move_history:
            for letter in move['word']:
                if letter in high_value_letters:
                    usage[letter] += 1
        
        return dict(usage)
    
    def _analyze_defensive_play(self) -> float:
        """Analyze tendency for defensive play"""
        if len(self.move_history) < 2:
            return 0.0
        
        # Simple heuristic: moves that don't maximize score might be defensive
        defensive_moves = 0
        for move in self.move_history:
            score = self._estimate_move_score(move)
            # If score is lower than expected for word length, might be defensive
            expected_score = len(move['word']) * 3  # Simple expectation
            if score < expected_score * 0.7:
                defensive_moves += 1
        
        return defensive_moves / len(self.move_history)
    
    def _estimate_move_score(self, move: Dict) -> int:
        """Estimate score of a move (simplified)"""
        # Simplified scoring - in practice, would calculate actual score
        word = move['word']
        base_score = len(word) * 2
        
        # Add bonus for high-value letters
        high_value = {'J': 8, 'Q': 10, 'X': 8, 'Z': 10, 'K': 5, 'V': 4, 'W': 4, 'Y': 4}
        for letter in word:
            base_score += high_value.get(letter, 0)
        
        return base_score