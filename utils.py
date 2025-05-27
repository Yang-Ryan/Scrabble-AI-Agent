"""
Utility functions for Scrabble AI
Helper functions for word validation, scoring, and game mechanics
"""

import json
import pickle
import random
from typing import List, Dict, Set, Tuple, Optional

# Scrabble tile values
TILE_VALUES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10, '?': 0  # Blank tile
}

# Standard Scrabble tile distribution
TILE_DISTRIBUTION = {
    'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
    'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
    'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
    'Y': 2, 'Z': 1, '?': 2  # Blank tiles
}

def load_dictionary(dictionary_path: str) -> Set[str]:
    """
    Load word dictionary from file
    
    Args:
        dictionary_path: Path to dictionary file
        
    Returns:
        Set of valid words (uppercase)
    """
    try:
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            words = set()
            for line in f:
                word = line.strip().upper()
                if word and len(word) >= 2:  # Only words 2+ letters
                    words.add(word)
        print(f"Loaded {len(words)} words from dictionary")
        return words
    except FileNotFoundError:
        print(f"Dictionary file not found: {dictionary_path}")
        # Return a small default dictionary for testing
        return {'CAT', 'DOG', 'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 
                'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD',
                'BY', 'WORD', 'WHAT', 'SAID', 'EACH', 'WHICH', 'DO', 'HOW',
                'THEIR', 'TIME', 'WILL', 'ABOUT', 'IF', 'UP', 'OUT', 'MANY'}

def is_valid_word(word: str, dictionary: Set[str]) -> bool:
    """
    Check if word is valid according to dictionary
    
    Args:
        word: Word to check (will be converted to uppercase)
        dictionary: Set of valid words
        
    Returns:
        True if word is valid
    """
    return word.upper() in dictionary

def calculate_word_score(word: str, positions: List[Tuple[int, int]], 
                        premium_squares: List[str], is_bingo: bool = False) -> int:
    """
    Calculate score for a word placement
    
    Args:
        word: The word being played
        positions: List of (row, col) positions
        premium_squares: List of premium square types used
        is_bingo: Whether all 7 tiles were used (50 point bonus)
        
    Returns:
        Total score for the word
    """
    base_score = 0
    word_multiplier = 1
    
    # Calculate base score with letter multipliers
    for i, letter in enumerate(word):
        letter_value = TILE_VALUES.get(letter, 0)
        letter_multiplier = 1
        
        # Apply premium square effects
        if i < len(premium_squares):
            premium = premium_squares[i]
            if premium == 'DL':  # Double Letter
                letter_multiplier = 2
            elif premium == 'TL':  # Triple Letter
                letter_multiplier = 3
            elif premium == 'DW':  # Double Word
                word_multiplier *= 2
            elif premium == 'TW':  # Triple Word
                word_multiplier *= 3
        
        base_score += letter_value * letter_multiplier
    
    # Apply word multiplier
    total_score = base_score * word_multiplier
    
    # Add bingo bonus
    if is_bingo:
        total_score += 50
    
    return total_score

def create_tile_bag() -> List[str]:
    """
    Create a shuffled bag of tiles according to standard Scrabble distribution
    
    Returns:
        List of tiles (shuffled)
    """
    tiles = []
    for letter, count in TILE_DISTRIBUTION.items():
        tiles.extend([letter] * count)
    
    random.shuffle(tiles)
    return tiles

def draw_tiles(tile_bag: List[str], num_tiles: int) -> List[str]:
    """
    Draw tiles from the bag
    
    Args:
        tile_bag: List of available tiles
        num_tiles: Number of tiles to draw
        
    Returns:
        List of drawn tiles
    """
    drawn = []
    for _ in range(min(num_tiles, len(tile_bag))):
        if tile_bag:
            drawn.append(tile_bag.pop())
    return drawn

def create_empty_board(size: int = 15) -> List[List[Optional[str]]]:
    """
    Create empty Scrabble board
    
    Args:
        size: Board size (default 15x15)
        
    Returns:
        2D list representing empty board
    """
    return [[None for _ in range(size)] for _ in range(size)]

def place_word_on_board(board: List[List[Optional[str]]], word: str, 
                       positions: List[Tuple[int, int]]) -> List[List[Optional[str]]]:
    """
    Place word on board and return new board state
    
    Args:
        board: Current board state
        word: Word to place
        positions: List of (row, col) positions
        
    Returns:
        New board state with word placed
    """
    new_board = [row.copy() for row in board]
    
    for i, (row, col) in enumerate(positions):
        if i < len(word):
            new_board[row][col] = word[i]
    
    return new_board

def get_rack_after_move(original_rack: List[str], tiles_used: List[str], 
                       tiles_drawn: List[str]) -> List[str]:
    """
    Calculate rack after making a move
    
    Args:
        original_rack: Tiles before move
        tiles_used: Tiles used in move
        tiles_drawn: New tiles drawn from bag
        
    Returns:
        New rack after move
    """
    new_rack = original_rack.copy()
    
    # Remove used tiles
    for tile in tiles_used:
        if tile in new_rack:
            new_rack.remove(tile)
    
    # Add newly drawn tiles
    new_rack.extend(tiles_drawn)
    
    return new_rack

def board_to_string(board: List[List[Optional[str]]]) -> str:
    """
    Convert board to string representation for display
    
    Args:
        board: Board state
        
    Returns:
        String representation of board
    """
    lines = []
    for row in board:
        line = ""
        for cell in row:
            line += cell if cell is not None else "."
        lines.append(line)
    return "\n".join(lines)

def save_game_data(data: Dict, filepath: str):
    """
    Save game data to file
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_game_data(filepath: str) -> Optional[Dict]:
    """
    Load game data from file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_rack_value(rack: List[str]) -> int:
    """
    Calculate total point value of tiles in rack
    
    Args:
        rack: List of tiles
        
    Returns:
        Total point value
    """
    return sum(TILE_VALUES.get(tile, 0) for tile in rack)

def get_vowel_consonant_ratio(rack: List[str]) -> float:
    """
    Calculate vowel to consonant ratio in rack
    
    Args:
        rack: List of tiles
        
    Returns:
        Ratio of vowels to total tiles
    """
    if not rack:
        return 0.0
    
    vowels = sum(1 for tile in rack if tile in 'AEIOU')
    return vowels / len(rack)

def simulate_random_rack(num_tiles: int = 7) -> List[str]:
    """
    Generate a random rack for testing
    
    Args:
        num_tiles: Number of tiles to generate
        
    Returns:
        Random rack
    """
    tile_bag = create_tile_bag()
    return draw_tiles(tile_bag, num_tiles)

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"

def get_letter_frequency(words: List[str]) -> Dict[str, int]:
    """
    Calculate letter frequency in list of words
    
    Args:
        words: List of words to analyze
        
    Returns:
        Dictionary of letter frequencies
    """
    freq = {}
    for word in words:
        for letter in word.upper():
            freq[letter] = freq.get(letter, 0) + 1
    return freq

def analyze_rack_potential(rack: List[str], dictionary: Set[str]) -> Dict[str, any]:
    """
    Analyze potential of a rack for word formation
    
    Args:
        rack: Tiles in rack
        dictionary: Valid words
        
    Returns:
        Analysis dictionary
    """
    analysis = {
        'total_value': calculate_rack_value(rack),
        'vowel_ratio': get_vowel_consonant_ratio(rack),
        'high_value_tiles': sum(1 for tile in rack if TILE_VALUES.get(tile, 0) >= 4),
        'blank_tiles': rack.count('?'),
        'duplicate_tiles': 0,
        'common_tiles': sum(1 for tile in rack if tile in 'ERSTAIN'),
        'bingo_potential': 0  # Could be calculated more sophisticated
    }
    
    # Count duplicates
    tile_counts = {}
    for tile in rack:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    analysis['duplicate_tiles'] = sum(1 for count in tile_counts.values() if count > 1)
    
    return analysis

def create_game_state(board: List[List[Optional[str]]], my_rack: List[str], 
                     opponent_rack: List[str], my_score: int, opponent_score: int,
                     tiles_remaining: int, rounds_played: int = 0) -> Dict:
    """
    Create standardized game state dictionary
    
    Args:
        board: Current board state
        my_rack: Player's tiles
        opponent_rack: Opponent's tiles (may be hidden)
        my_score: Player's current score
        opponent_score: Opponent's current score
        tiles_remaining: Tiles left in bag
        rounds_played: Number of rounds completed
        
    Returns:
        Game state dictionary
    """
    return {
        'board': board,
        'my_rack': my_rack,
        'opponent_rack': opponent_rack,
        'my_score': my_score,
        'opponent_score': opponent_score,
        'tiles_remaining': tiles_remaining,
        'rounds_played': rounds_played,
        'score_gap': my_score - opponent_score
    }

def validate_rack(rack: List[str]) -> bool:
    """
    Validate that rack contains only legal tiles
    
    Args:
        rack: Tiles to validate
        
    Returns:
        True if rack is valid
    """
    if len(rack) > 7:
        return False
    
    for tile in rack:
        if tile not in TILE_VALUES:
            return False
    
    return True

def get_adjacent_positions(row: int, col: int, board_size: int = 15) -> List[Tuple[int, int]]:
    """
    Get adjacent positions on board
    
    Args:
        row: Current row
        col: Current column  
        board_size: Size of board
        
    Returns:
        List of adjacent (row, col) positions
    """
    adjacent = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < board_size and 0 <= new_col < board_size:
            adjacent.append((new_row, new_col))
    return adjacent

def count_empty_adjacent_squares(board: List[List[Optional[str]]], 
                                positions: List[Tuple[int, int]]) -> int:
    """
    Count empty squares adjacent to given positions
    
    Args:
        board: Current board state
        positions: Positions to check around
        
    Returns:
        Number of empty adjacent squares
    """
    empty_adjacent = set()
    
    for row, col in positions:
        for adj_row, adj_col in get_adjacent_positions(row, col):
            if board[adj_row][adj_col] is None:
                empty_adjacent.add((adj_row, adj_col))
    
    return len(empty_adjacent)

def estimate_tiles_remaining_by_letter(played_tiles: List[str]) -> Dict[str, int]:
    """
    Estimate remaining tiles in bag based on played tiles
    
    Args:
        played_tiles: All tiles that have been played
        
    Returns:
        Dictionary of estimated remaining tiles by letter
    """
    remaining = TILE_DISTRIBUTION.copy()
    
    for tile in played_tiles:
        if tile in remaining:
            remaining[tile] = max(0, remaining[tile] - 1)
    
    return remaining

def calculate_endgame_penalty(rack: List[str]) -> int:
    """
    Calculate penalty for tiles left in rack at game end
    
    Args:
        rack: Remaining tiles
        
    Returns:
        Penalty points
    """
    return sum(TILE_VALUES.get(tile, 0) for tile in rack)

def is_rack_stuck(rack: List[str], dictionary: Set[str], 
                 board: List[List[Optional[str]]]) -> bool:
    """
    Check if rack is "stuck" (cannot form any valid words)
    Simplified version - in reality would check against board constraints
    
    Args:
        rack: Current rack
        dictionary: Valid words
        board: Current board state
        
    Returns:
        True if rack appears stuck
    """
    if not rack:
        return True
    
    # Check if any 2-3 letter words can be formed
    for word in dictionary:
        if 2 <= len(word) <= 3:
            if can_form_word_from_rack(word, rack):
                return False
    
    return True

def can_form_word_from_rack(word: str, rack: List[str]) -> bool:
    """
    Check if word can be formed from tiles in rack
    
    Args:
        word: Word to check
        rack: Available tiles
        
    Returns:
        True if word can be formed
    """
    available = rack.copy()
    
    for letter in word.upper():
        if letter in available:
            available.remove(letter)
        elif '?' in available:  # Use blank tile
            available.remove('?')
        else:
            return False
    
    return True

def generate_summary_stats(games_data: List[Dict]) -> Dict:
    """
    Generate summary statistics from multiple games
    
    Args:
        games_data: List of game result dictionaries
        
    Returns:
        Summary statistics
    """
    if not games_data:
        return {}
    
    scores = [game.get('final_score', 0) for game in games_data]
    score_gaps = [game.get('final_score_gap', 0) for game in games_data]
    game_lengths = [game.get('total_moves', 0) for game in games_data]
    
    stats = {
        'total_games': len(games_data),
        'average_score': sum(scores) / len(scores),
        'average_score_gap': sum(score_gaps) / len(score_gaps),
        'average_game_length': sum(game_lengths) / len(game_lengths),
        'win_rate': sum(1 for gap in score_gaps if gap > 0) / len(score_gaps),
        'max_score': max(scores),
        'min_score': min(scores),
        'max_score_gap': max(score_gaps),
        'min_score_gap': min(score_gaps)
    }
    
    return stats

def print_board(board: List[List[Optional[str]]], show_coordinates: bool = False):
    """
    Print board to console in readable format
    
    Args:
        board: Board state to print
        show_coordinates: Whether to show row/column numbers
    """
    if show_coordinates:
        print("   " + " ".join(f"{i:2d}" for i in range(len(board[0]))))
    
    for i, row in enumerate(board):
        line = f"{i:2d} " if show_coordinates else ""
        for cell in row:
            line += (cell if cell is not None else '.') + ' '
        print(line)

def create_move_dict(word: str, positions: List[Tuple[int, int]], 
                    direction: str, tiles_used: List[str], 
                    remaining_rack: List[str], score: int = 0) -> Dict:
    """
    Create standardized move dictionary
    
    Args:
        word: Word being played
        positions: Board positions
        direction: 'horizontal' or 'vertical'
        tiles_used: Tiles used from rack
        remaining_rack: Tiles left in rack after move
        score: Points scored
        
    Returns:
        Move dictionary
    """
    return {
        'word': word,
        'positions': positions,
        'direction': direction,
        'tiles_used': tiles_used,
        'remaining_rack': remaining_rack,
        'score': score,
        'word_length': len(word),
        'position': positions[0] if positions else (0, 0)  # Starting position
    }