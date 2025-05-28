"""
Fixed Move Generator for Scrabble AI
Simple but working implementation that finds valid moves
"""

import random
from typing import List, Dict, Set, Tuple, Optional
from utils import load_dictionary, is_valid_word

class MoveGenerator:
    """
    Working move generator - replaces the buggy complex version
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        self.dictionary = load_dictionary(dictionary_path)
        
        # Create smaller working set for better performance
        self.working_words = []
        for word in self.dictionary:
            if 2 <= len(word) <= 7:  # Reasonable word lengths
                self.working_words.append(word)
        
        print(f"MoveGenerator: Using {len(self.working_words)} words for move generation")
        
        # Letter values for scoring
        self.letter_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, '?': 0
        }
    
    def get_valid_moves(self, board: List[List[str]], rack: List[str], 
                       sampling_rate: float = 1.0) -> List[Dict]:
        """
        Find valid moves for current board and rack
        """
        if self._is_empty_board(board):
            moves = self._get_opening_moves(rack)
        else:
            moves = self._get_subsequent_moves(board, rack)
        
        # Apply sampling for training speed
        if sampling_rate < 1.0 and len(moves) > 10:
            sample_size = max(5, int(len(moves) * sampling_rate))
            moves = random.sample(moves, sample_size)
        
        return moves
    
    def _is_empty_board(self, board: List[List[str]]) -> bool:
        """Check if board is completely empty"""
        for row in board:
            for cell in row:
                if cell is not None and cell != '':
                    return False
        return True
    
    def _get_opening_moves(self, rack: List[str]) -> List[Dict]:
        """Generate opening moves (must go through center)"""
        moves = []
        center_row, center_col = 7, 7
        
        # Try words that can be formed from rack
        possible_words = self._find_formable_words(rack)
        
        for word in possible_words[:30]:  # Limit for performance
            # Try horizontal placement through center
            word_len = len(word)
            
            # Horizontal - center the word on center square
            start_col = max(0, center_col - word_len // 2)
            if start_col + word_len <= 15:  # Fits on board
                positions = [(center_row, start_col + i) for i in range(word_len)]
                
                # Make sure center square is used
                if (center_row, center_col) in positions:
                    move = self._create_move_dict(word, positions, 'horizontal', rack)
                    if move:
                        moves.append(move)
            
            # Vertical - center the word on center square  
            start_row = max(0, center_row - word_len // 2)
            if start_row + word_len <= 15:  # Fits on board
                positions = [(start_row + i, center_col) for i in range(word_len)]
                
                # Make sure center square is used
                if (center_row, center_col) in positions:
                    move = self._create_move_dict(word, positions, 'vertical', rack)
                    if move:
                        moves.append(move)
        
        return moves
    
    def _get_subsequent_moves(self, board: List[List[str]], rack: List[str]) -> List[Dict]:
        """Generate moves that connect to existing tiles"""
        moves = []
        
        # Find anchor points (empty squares next to filled squares)
        anchor_points = self._find_anchor_points(board)
        
        possible_words = self._find_formable_words(rack)
        
        # Try placing words at anchor points
        for word in possible_words[:20]:  # Limit for performance
            for anchor_row, anchor_col in anchor_points[:8]:  # Try first 8 anchors
                
                # Try horizontal placement starting at anchor
                if anchor_col + len(word) <= 15:
                    positions = [(anchor_row, anchor_col + i) for i in range(len(word))]
                    if self._positions_are_valid(board, positions):
                        move = self._create_move_dict(word, positions, 'horizontal', rack)
                        if move:
                            moves.append(move)
                
                # Try vertical placement starting at anchor
                if anchor_row + len(word) <= 15:
                    positions = [(anchor_row + i, anchor_col) for i in range(len(word))]
                    if self._positions_are_valid(board, positions):
                        move = self._create_move_dict(word, positions, 'vertical', rack)
                        if move:
                            moves.append(move)
        
        return moves
    
    def _find_anchor_points(self, board: List[List[str]]) -> List[Tuple[int, int]]:
        """Find empty squares adjacent to filled squares"""
        anchors = set()
        
        for row in range(15):
            for col in range(15):
                if board[row][col] is not None and board[row][col] != '':
                    # Check adjacent squares
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < 15 and 0 <= new_col < 15 and
                            (board[new_row][new_col] is None or board[new_row][new_col] == '')):
                            anchors.add((new_row, new_col))
        
        return list(anchors)
    
    def _positions_are_valid(self, board: List[List[str]], positions: List[Tuple[int, int]]) -> bool:
        """Check if all positions are empty and in bounds"""
        for row, col in positions:
            if (row < 0 or row >= 15 or col < 0 or col >= 15):
                return False
            if board[row][col] is not None and board[row][col] != '':
                return False
        return True
    
    def _find_formable_words(self, rack: List[str]) -> List[str]:
        """Find words that can be formed from the given rack"""
        formable = []
        
        for word in self.working_words:
            if self._can_form_word(word, rack):
                formable.append(word)
        
        # Sort by length and score potential
        formable.sort(key=lambda w: (len(w), self._calculate_word_value(w)), reverse=True)
        
        return formable
    
    def _can_form_word(self, word: str, rack: List[str]) -> bool:
        """Check if word can be formed from rack tiles"""
        available_tiles = rack.copy()
        
        for letter in word.upper():
            if letter in available_tiles:
                available_tiles.remove(letter)
            elif '?' in available_tiles:  # Use blank tile
                available_tiles.remove('?')
            else:
                return False
        
        return True
    
    def _calculate_word_value(self, word: str) -> int:
        """Calculate basic point value of word"""
        return sum(self.letter_values.get(letter, 1) for letter in word.upper())
    
    def _create_move_dict(self, word: str, positions: List[Tuple[int, int]], 
                         direction: str, original_rack: List[str]) -> Optional[Dict]:
        """Create standardized move dictionary"""
        
        # Calculate tiles used from rack
        tiles_used = []
        remaining_rack = original_rack.copy()
        
        for letter in word.upper():
            if letter in remaining_rack:
                remaining_rack.remove(letter)
                tiles_used.append(letter)
            elif '?' in remaining_rack:
                remaining_rack.remove('?')
                tiles_used.append('?')  # Blank tile used as this letter
        
        if len(tiles_used) != len(word):
            return None  # Couldn't form word properly
        
        # Calculate score
        base_score = self._calculate_word_value(word)
        
        # Bonus for long words
        if len(word) == 7:
            base_score += 50  # Bingo bonus
        elif len(word) >= 5:
            base_score += 10  # Length bonus
        
        # Premium squares (simplified)
        premium_squares_used = []
        if (7, 7) in positions:  # Center square
            premium_squares_used.append('DW')
            base_score *= 2
        
        # Add some randomness to make interesting
        if len(word) >= 4:
            base_score += random.randint(0, 5)
        
        move = {
            'word': word.upper(),
            'positions': positions,
            'direction': direction,
            'tiles_used': tiles_used,
            'remaining_rack': remaining_rack,
            'score': base_score,
            'premium_squares_used': premium_squares_used,
            'word_length': len(word),
            'position': positions[0] if positions else (0, 0),
            
            # Additional info for feature extraction
            'adjacent_empty_squares': len(positions) * 2,  # Approximation
            'premium_squares_blocked': 0,  # Simplified
            'word_hooks_created': 1 if len(word) >= 4 else 0,
            'board_congestion_increase': len(word) // 3
        }
        
        return move


# Test function
def test_move_generator():
    """Test the move generator"""
    print("Testing MoveGenerator...")
    
    mg = MoveGenerator()
    
    # Test with empty board
    from utils import create_empty_board
    board = create_empty_board()
    test_rack = ['C', 'A', 'R', 'E', 'S', 'T', 'N']
    
    moves = mg.get_valid_moves(board, test_rack)
    print(f"Found {len(moves)} moves for rack {test_rack}")
    
    if moves:
        print("Sample moves:")
        for i, move in enumerate(moves[:5]):
            print(f"  {i+1}. {move['word']} - {move['score']} points")
    else:
        print("❌ No moves found!")

if __name__ == "__main__":
    test_move_generator()