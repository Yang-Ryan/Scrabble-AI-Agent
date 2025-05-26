import random
import numpy as np
from typing import Dict, List, Optional
from scrabble_game import Player, ScrabbleGame

class RandomAgent(Player):
    """Baseline agent that makes random valid moves"""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get random valid move"""
        valid_moves = game.get_valid_moves(self.tiles)
        
        if not valid_moves:
            return None  # Pass turn
        
        return random.choice(valid_moves)

class GreedyAgent(Player):
    """Baseline agent that always picks the highest scoring move"""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get highest scoring valid move"""
        valid_moves = game.get_valid_moves(self.tiles)
        
        if not valid_moves:
            return None  # Pass turn
        
        # Return move with highest score
        return max(valid_moves, key=lambda move: move['score'])

class HeuristicAgent(Player):
    """More sophisticated baseline using heuristics"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.tile_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, '_': 0
        }
    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get move using heuristic evaluation"""
        valid_moves = game.get_valid_moves(self.tiles)
        
        if not valid_moves:
            return None
        
        # Evaluate each move with heuristics
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            heuristic_score = self._evaluate_move(move, game)
            if heuristic_score > best_score:
                best_score = heuristic_score
                best_move = move
        
        return best_move
    
    def _evaluate_move(self, move: Dict, game: ScrabbleGame) -> float:
        """Evaluate move using multiple heuristics"""
        score = move['score']
        word = move['word']
        position = move['position']
        
        # Base score
        evaluation = score
        
        # Word length bonus
        if len(word) >= 7:
            evaluation += 50  # Bingo bonus
        elif len(word) >= 5:
            evaluation += 10
        
        # High-value letter bonus
        high_value_letters = {'J', 'Q', 'X', 'Z'}
        for letter in word:
            if letter in high_value_letters:
                evaluation += 15
        
        # Tile management heuristic
        evaluation += self._tile_management_score(word)
        
        # Board position heuristic
        evaluation += self._position_score(position, game)
        
        # Opening strategy
        if game.turn_count < 3:
            evaluation += self._opening_strategy_score(move, game)
        
        # Endgame strategy
        if game.tile_bag.remaining_count() < 15:
            evaluation += self._endgame_strategy_score(move)
        
        return evaluation
    
    def _tile_management_score(self, word: str) -> float:
        """Score based on tile management principles"""
        score = 0
        
        # Vowel-consonant balance after move
        used_tiles = list(word)
        remaining_tiles = self.tiles.copy()
        for tile in used_tiles:
            if tile in remaining_tiles:
                remaining_tiles.remove(tile)
        
        vowels = sum(1 for tile in remaining_tiles if tile in 'AEIOU')
        consonants = len(remaining_tiles) - vowels
        
        # Penalty for poor balance
        if vowels > 5 or consonants > 5:
            score -= 10
        
        # Bonus for good balance
        if 2 <= vowels <= 4 and 2 <= consonants <= 4:
            score += 5
        
        # Bonus for using difficult tiles
        difficult_tiles = {'Q', 'X', 'Z', 'J'}
        for letter in word:
            if letter in difficult_tiles:
                score += 10
        
        return score
    
    def _position_score(self, position: tuple[int, int], game: ScrabbleGame) -> float:
        """Score based on board position"""
        score = 0
        row, col = position
        
        # Bonus for playing near center early in game
        if game.turn_count < 5:
            center_distance = abs(row - 7) + abs(col - 7)
            score += max(0, 10 - center_distance)
        
        # Check if position opens premium squares for opponent
        premium_squares = game.board.premium_squares
        
        # Penalty for creating opportunities near premium squares
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_pos = (row + dr, col + dc)
                if new_pos in premium_squares and game.board.board[new_pos[0]][new_pos[1]] == '':
                    score -= 3  # Small penalty for creating opportunities
        
        return score
    
    def _opening_strategy_score(self, move: Dict, game: ScrabbleGame) -> float:
        """Score for opening moves"""
        score = 0
        word = move['word']
        
        # Prefer balanced words in opening
        vowels = sum(1 for letter in word if letter in 'AEIOU')
        consonants = len(word) - vowels
        
        if 0.3 <= vowels / len(word) <= 0.6:  # Good balance
            score += 8
        
        # Prefer medium-length words early
        if 4 <= len(word) <= 6:
            score += 5
        
        return score
    
    def _endgame_strategy_score(self, move: Dict) -> float:
        """Score for endgame moves"""
        score = 0
        word = move['word']
        
        # Bonus for using more tiles
        score += len(word) * 3
        
        # Bonus for high-point tiles
        for letter in word:
            if self.tile_values.get(letter, 0) >= 4:
                score += 5
        
        return score

class MinimaxAgent(Player):
    """Minimax agent with limited depth for baseline comparison"""
    
    def __init__(self, name: str, depth: int = 2):
        super().__init__(name)
        self.depth = depth
        self.tile_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, '_': 0
        }
    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get move using minimax search"""
        valid_moves = game.get_valid_moves(self.tiles)
        
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Use minimax with alpha-beta pruning
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            # Simulate move
            game_copy = self._simulate_move(game, move, self)
            value = self._minimax(game_copy, self.depth - 1, False, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Alpha-beta pruning
        
        return best_move
    
    def _minimax(self, game: ScrabbleGame, depth: int, maximizing: bool, alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0 or game.game_over:
            return self._evaluate_position(game)
        
        if maximizing:
            max_eval = float('-inf')
            valid_moves = game.get_valid_moves(self.tiles)
            
            if not valid_moves:
                return self._evaluate_position(game)
            
            for move in valid_moves[:5]:  # Limit branching factor
                game_copy = self._simulate_move(game, move, self)
                eval_score = self._minimax(game_copy, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            # Get opponent's moves (simplified)
            opponent = next(p for p in game.players if p != self)
            valid_moves = game.get_valid_moves(opponent.tiles)
            
            if not valid_moves:
                return self._evaluate_position(game)
            
            for move in valid_moves[:5]:  # Limit branching factor
                game_copy = self._simulate_move(game, move, opponent)
                eval_score = self._minimax(game_copy, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval
    
    def _simulate_move(self, game: ScrabbleGame, move: Dict, player: Player):
        """Create a copy of game with move applied"""
        # This is a simplified simulation
        # In practice, you'd create a deep copy and apply the move
        return game  # Placeholder - would need full implementation
    
    def _evaluate_position(self, game: ScrabbleGame) -> float:
        """Evaluate current position"""
        # Simple evaluation: score difference
        my_score = self.score
        opponent_scores = [p.score for p in game.players if p != self]
        opponent_score = max(opponent_scores) if opponent_scores else 0
        
        return my_score - opponent_score

class AdaptiveAgent(Player):
    """Agent that adapts strategy based on game state"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.strategy_weights = {
            'greedy': 0.4,
            'defensive': 0.3,
            'tile_management': 0.2,
            'positional': 0.1
        }
    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get move using adaptive strategy"""
        valid_moves = game.get_valid_moves(self.tiles)
        
        if not valid_moves:
            return None
        
        # Adapt strategy based on game state
        self._adapt_strategy(game)
        
        # Evaluate moves with current strategy
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            score = self._evaluate_adaptive_move(move, game)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _adapt_strategy(self, game: ScrabbleGame):
        """Adapt strategy weights based on game state"""
        tiles_remaining = game.tile_bag.remaining_count()
        turn_count = game.turn_count
        
        # Early game: focus on tile management and position
        if tiles_remaining > 70:
            self.strategy_weights = {
                'greedy': 0.3,
                'defensive': 0.2,
                'tile_management': 0.3,
                'positional': 0.2
            }
        # Mid game: balanced approach
        elif tiles_remaining > 30:
            self.strategy_weights = {
                'greedy': 0.4,
                'defensive': 0.3,
                'tile_management': 0.2,
                'positional': 0.1
            }
        # End game: focus on scoring and defense
        else:
            self.strategy_weights = {
                'greedy': 0.5,
                'defensive': 0.4,
                'tile_management': 0.1,
                'positional': 0.0
            }
        
        # Adjust based on score difference
        my_score = self.score
        opponent_scores = [p.score for p in game.players if p != self]
        if opponent_scores:
            max_opponent_score = max(opponent_scores)
            if my_score < max_opponent_score - 50:  # Behind
                self.strategy_weights['greedy'] += 0.2
                self.strategy_weights['defensive'] -= 0.1
            elif my_score > max_opponent_score + 50:  # Ahead
                self.strategy_weights['defensive'] += 0.2
                self.strategy_weights['greedy'] -= 0.1
    
    def _evaluate_adaptive_move(self, move: Dict, game: ScrabbleGame) -> float:
        """Evaluate move using adaptive weights"""
        scores = {
            'greedy': move['score'],
            'defensive': self._defensive_score(move, game),
            'tile_management': self._tile_management_score(move),
            'positional': self._positional_score(move, game)
        }
        
        # Weighted combination
        total_score = sum(
            self.strategy_weights[strategy] * score
            for strategy, score in scores.items()
        )
        
        return total_score
    
    def _defensive_score(self, move: Dict, game: ScrabbleGame) -> float:
        """Calculate defensive value of move"""
        # Simplified defensive scoring
        position = move['position']
        
        # Check if move blocks premium squares
        premium_squares = game.board.premium_squares
        blocking_score = 0
        
        row, col = position
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_pos = (row + dr, col + dc)
                if new_pos in premium_squares:
                    blocking_score += 5
        
        return blocking_score
    
    def _tile_management_score(self, move: Dict) -> float:
        """Calculate tile management value of move"""
        word = move['word']
        score = 0
        
        # Simulate tiles after move
        used_tiles = list(word)
        remaining_tiles = self.tiles.copy()
        for tile in used_tiles:
            if tile in remaining_tiles:
                remaining_tiles.remove(tile)
        
        # Vowel-consonant balance
        vowels = sum(1 for tile in remaining_tiles if tile in 'AEIOU')
        consonants = len(remaining_tiles) - vowels
        
        # Good balance bonus
        if len(remaining_tiles) > 0:
            vowel_ratio = vowels / len(remaining_tiles)
            if 0.3 <= vowel_ratio <= 0.6:
                score += 10
        
        # Penalty for keeping difficult tiles
        difficult_tiles = {'Q', 'X', 'Z', 'J'}
        for letter in word:
            if letter in difficult_tiles:
                score += 15  # Bonus for using difficult tiles
        
        return score
    
    def _positional_score(self, move: Dict, game: ScrabbleGame) -> float:
        """Calculate positional value of move"""
        position = move['position']
        row, col = position
        
        # Distance from center bonus (early game)
        if game.turn_count < 5:
            center_distance = abs(row - 7) + abs(col - 7)
            return max(0, 15 - center_distance)
        
        # Later game: avoid creating opportunities
        return 0