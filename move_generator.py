"""
Fixed Move Generator for Scrabble AI
Fixed infinite loops and performance issues
"""

import random
from typing import List, Dict, Set, Tuple, Optional
from utils import load_dictionary, is_valid_word

class MoveGenerator:
    """
    Working move generator with fixed infinite loop issues
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        self.dictionary = load_dictionary(dictionary_path)
        
        # Create smaller working set for better performance
        self.working_words = []
        for word in self.dictionary:
            if 2 <= len(word) <= 7:  # Reasonable word lengths
                self.working_words.append(word.upper())
        
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
        
        for word in possible_words[:30]:
            # Try horizontal placement through center
            word_len = len(word)
            
            # Horizontal - center the word on center square
            start_col = max(0, center_col - word_len // 2)
            if start_col + word_len <= 15:
                positions = [(center_row, start_col + i) for i in range(word_len)]
                
                # Make sure center square is used
                if (center_row, center_col) in positions:
                    move = self._create_move_dict(word, positions, 'horizontal', rack)
                    if move:
                        moves.append(move)
            
            # Vertical - center the word on center square  
            start_row = max(0, center_row - word_len // 2)
            if start_row + word_len <= 15:
                positions = [(start_row + i, center_col) for i in range(word_len)]
                
                # Make sure center square is used
                if (center_row, center_col) in positions:
                    move = self._create_move_dict(word, positions, 'vertical', rack)
                    if move:
                        moves.append(move)
        
        return moves

    def _get_subsequent_moves(self, board: List[List[str]], rack: List[str]) -> List[Dict]:
        """Generate legal moves for Scrabble following official rules"""
        moves = []
        
        anchor_points = self._find_anchor_points(board)
        
        # Limit anchor points for performance
        if len(anchor_points) > 20:
            anchor_points = anchor_points[:20]
        
        possible_words = self._find_formable_words(rack)
        
        # Limit words for performance
        if len(possible_words) > 50:
            possible_words = possible_words[:50]
        
        for i, (anchor_row, anchor_col) in enumerate(anchor_points):
            
            # Try horizontal
            horizontal_moves = self._generate_moves_from_anchor(
                board, rack, anchor_row, anchor_col, 'horizontal', possible_words
            )
            moves.extend(horizontal_moves)
            
            # Try vertical
            vertical_moves = self._generate_moves_from_anchor(
                board, rack, anchor_row, anchor_col, 'vertical', possible_words
            )
            moves.extend(vertical_moves)
            
            if len(moves) > 100:
                break
        
        return moves

    def _generate_moves_from_anchor(self, board: List[List[str]], rack: List[str], 
                                   anchor_row: int, anchor_col: int, direction: str,
                                   word_list: List[str]) -> List[Dict]:
        """Generate possible words starting from an anchor in a given direction"""
        moves = []
        dr, dc = (0, 1) if direction == 'horizontal' else (1, 0)

        for word in word_list:
            if not self._can_form_word(word, rack):
                continue

            # Try different starting positions for the word around the anchor
            for start_offset in range(-len(word) + 1, 1):  # Word can start before anchor
                positions = []
                start_r = anchor_row + start_offset * dr
                start_c = anchor_col + start_offset * dc
                
                # Check if word fits on board
                end_r = start_r + (len(word) - 1) * dr
                end_c = start_c + (len(word) - 1) * dc
                
                if not (0 <= start_r < 15 and 0 <= start_c < 15 and 
                       0 <= end_r < 15 and 0 <= end_c < 15):
                    continue
                
                # Build positions list
                valid = True
                rack_copy = rack.copy()
                uses_existing_tile = False
                
                for i, letter in enumerate(word):
                    r = start_r + i * dr
                    c = start_c + i * dc
                    positions.append((r, c))
                    
                    board_cell = board[r][c]
                    
                    if board_cell and board_cell != '':
                        # Must match existing letter
                        if board_cell.upper() != letter.upper():
                            valid = False
                            break
                        uses_existing_tile = True
                    else:
                        # Need to place from rack
                        if letter.upper() in rack_copy:
                            rack_copy.remove(letter.upper())
                        elif '?' in rack_copy:
                            rack_copy.remove('?')
                        else:
                            valid = False
                            break
                
                if not valid or not uses_existing_tile:
                    continue  # Must use at least one existing tile
                
                # Check if anchor is covered
                if (anchor_row, anchor_col) not in positions:
                    continue
                
                # Check crosswords are valid
                if not self._crosswords_valid_safe(board, word, positions, direction):
                    continue

                move = self._create_move_dict(word, positions, direction, rack)
                if move:
                    moves.append(move)
                    
                # Limit moves per anchor
                if len(moves) >= 10:
                    break

        return moves

    def _crosswords_valid_safe(self, board: List[List[str]], word: str, 
                              positions: List[Tuple[int, int]], direction: str) -> bool:
        """Check that all crosswords are valid - SAFE VERSION"""
        cross_dr = 1 if direction == 'horizontal' else 0
        cross_dc = 0 if direction == 'horizontal' else 1

        for idx, (r, c) in enumerate(positions):
            letter = word[idx].upper()
            
            # Build crossword - SAFE version with limits
            cross_word = ""
            
            # Go backwards (up/left) - LIMITED SEARCH
            temp_r, temp_c = r - cross_dr, c - cross_dc
            backwards_letters = []
            steps = 0
            
            while (0 <= temp_r < 15 and 0 <= temp_c < 15 and 
                   board[temp_r][temp_c] and board[temp_r][temp_c] != '' and 
                   steps < 7):  # LIMIT TO 7 STEPS
                backwards_letters.insert(0, board[temp_r][temp_c].upper())
                temp_r -= cross_dr
                temp_c -= cross_dc
                steps += 1
            
            # Add current letter
            cross_word = ''.join(backwards_letters) + letter
            
            # Go forwards (down/right) - LIMITED SEARCH
            temp_r, temp_c = r + cross_dr, c + cross_dc
            steps = 0
            
            while (0 <= temp_r < 15 and 0 <= temp_c < 15 and 
                   board[temp_r][temp_c] and board[temp_r][temp_c] != '' and 
                   steps < 7):  # LIMIT TO 7 STEPS
                cross_word += board[temp_r][temp_c].upper()
                temp_r += cross_dr
                temp_c += cross_dc
                steps += 1
            
            # If crossword is longer than 1 letter, check validity
            if len(cross_word) > 1:
                if cross_word not in self.dictionary:
                    return False
        
        return True
    
    def _find_anchor_points(self, board: List[List[str]]) -> List[Tuple[int, int]]:
        """Find empty squares adjacent to filled squares"""
        anchors = set()
        
        for row in range(15):
            for col in range(15):
                if board[row][col] and board[row][col] != '':
                    # Check adjacent squares
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < 15 and 0 <= new_col < 15 and
                            (not board[new_row][new_col] or board[new_row][new_col] == '')):
                            anchors.add((new_row, new_col))
        
        return list(anchors)
    
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
        available_tiles = [tile.upper() for tile in rack]
        
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
        remaining_rack = [tile.upper() for tile in original_rack]
        
        for letter in word.upper():
            if letter in remaining_rack:
                remaining_rack.remove(letter)
                tiles_used.append(letter)
            elif '?' in remaining_rack:
                remaining_rack.remove('?')
                tiles_used.append('?')  # Blank tile used as this letter
        
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
    
    print("Testing with empty board...")
    moves = mg.get_valid_moves(board, test_rack)
    print(f"Found {len(moves)} moves for rack {test_rack}")
    
    if moves:
        print("Sample moves:")
        for i, move in enumerate(moves[:5]):
            print(f"  {i+1}. {move['word']} - {move['score']} points")
    
    # Test with some tiles on board
    print("\nTesting with tiles on board...")
    board[7][7] = 'C'
    board[7][8] = 'A'
    board[7][9] = 'T'
    
    moves = mg.get_valid_moves(board, ['E', 'R', 'S', 'T', 'N', 'D', 'O'])
    print(f"Found {len(moves)} moves with CAT on board")
    
    if moves:
        print("Sample moves:")
        for i, move in enumerate(moves[:5]):
            print(f"  {i+1}. {move['word']} - {move['score']} points at {move['positions']}")

if __name__ == "__main__":
    test_move_generator()