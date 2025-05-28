import random
import numpy as np
from trie import Trie
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict

class TileBag:
    def __init__(self):
        self.letter_distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3,
            'H': 2, 'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6,
            'O': 8, 'P': 2, 'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4,
            'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, '_': 2
        }
        self.tiles = []
        for letter, count in self.letter_distribution.items():
            self.tiles.extend([letter] * count)
        random.shuffle(self.tiles)

    def draw_tiles(self, num):
        drawn = []
        for _ in range(num):
            if self.tiles:
                drawn.append(self.tiles.pop())
        return drawn

    def return_tiles(self, tiles):
        self.tiles.extend(tiles)
        random.shuffle(self.tiles)

    def remaining_count(self):
        return len(self.tiles)

    def tile_counter(self):
        return Counter(self.tiles)

class ScrabbleBoard:
    """Scrabble board representation with premium squares"""
    
    def __init__(self, size=15):
        self.size = size
        self.board = [['' for _ in range(size)] for _ in range(size)]
        self.premium_squares = self._initialize_premium_squares()
        
    def _initialize_premium_squares(self):
        """Initialize premium squares (Triple Word, Double Word, etc.)"""
        premium = {}
        # Triple Word Score
        tws_positions = [(0,0), (0,7), (0,14), (7,0), (7,14), (14,0), (14,7), (14,14)]
        for pos in tws_positions:
            premium[pos] = 'TWS'
            
        # Double Word Score  
        dws_positions = [(1,1), (2,2), (3,3), (4,4), (1,13), (2,12), (3,11), (4,10),
                        (13,1), (12,2), (11,3), (10,4), (13,13), (12,12), (11,11), (10,10)]
        for pos in dws_positions:
            premium[pos] = 'DWS'
            
        # Triple Letter Score
        tls_positions = [(1,5), (1,9), (5,1), (5,5), (5,9), (5,13), (9,1), (9,5), 
                        (9,9), (9,13), (13,5), (13,9)]
        for pos in tls_positions:
            premium[pos] = 'TLS'
            
        # Double Letter Score
        dls_positions = [(0,3), (0,11), (2,6), (2,8), (3,0), (3,7), (3,14), (6,2),
                        (6,6), (6,8), (6,12), (7,3), (7,11), (8,2), (8,6), (8,8), 
                        (8,12), (11,0), (11,7), (11,14), (12,6), (12,8), (14,3), (14,11)]
        for pos in dls_positions:
            premium[pos] = 'DLS'
            
        return premium
    
    def place_word(self, word: str, start_pos: Tuple[int, int], direction: str) -> bool:
        """Place a word on the board"""
        if not self.is_valid_placement(word, start_pos, direction):
            return False
            
        row, col = start_pos
        for i, letter in enumerate(word):
            if direction == 'horizontal':
                if self.board[row][col + i] == '':
                    self.board[row][col + i] = letter
            else:  # vertical
                if self.board[row + i][col] == '':
                    self.board[row + i][col] = letter
        return True
    
    def is_valid_placement(self, word: str, start_pos: Tuple[int, int], direction: str) -> bool:
        """Check if word placement is valid"""
        row, col = start_pos
        
        # Check bounds
        if direction == 'horizontal':
            if col + len(word) > self.size:
                return False
        else:  # vertical
            if row + len(word) > self.size:
                return False
        
        # Check if placement conflicts with existing letters
        for i, letter in enumerate(word):
            if direction == 'horizontal':
                current = self.board[row][col + i]
            else:
                current = self.board[row + i][col]
                
            if current != '' and current != letter:
                return False
        
        return True
    
    def get_adjacent_words(self, word: str, start_pos: Tuple[int, int], direction: str) -> List[str]:
        """Get all words formed by placing this word"""
        formed_words = [word]
        # Implementation for finding cross words would go here
        return formed_words
    
    def copy(self):
        """Create a copy of the board"""
        new_board = ScrabbleBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        return new_board

class TileBag:
    """Scrabble tile bag with letter distribution"""
    
    def __init__(self):
        # Standard Scrabble letter distribution
        self.letter_distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
            'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
            'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
            'Y': 2, 'Z': 1, '_': 2  # Blank tiles
        }
        
        self.letter_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, '_': 0
        }
        
        self.tiles = []
        self._initialize_bag()
    
    def _initialize_bag(self):
        """Initialize the tile bag"""
        for letter, count in self.letter_distribution.items():
            self.tiles.extend([letter] * count)
        random.shuffle(self.tiles)
    
    def draw_tiles(self, count: int) -> List[str]:
        """Draw tiles from the bag"""
        drawn = []
        for _ in range(min(count, len(self.tiles))):
            if self.tiles:
                drawn.append(self.tiles.pop())
        return drawn
    
    def remaining_count(self) -> int:
        """Get remaining tile count"""
        return len(self.tiles)

class ScrabbleGame:
    """Main Scrabble game class"""
    
    def __init__(self, dictionary_file: Optional[str] = None):
        self.board = ScrabbleBoard()
        self.tile_bag = TileBag()
        self.players = []
        self.current_player = 0
        self.game_over = False
        self.turn_count = 0
        
        # Load dictionary
        self.trie = Trie()
        self.dictionary = self._load_dictionary(dictionary_file)
        
    def _load_dictionary(self, dictionary_file: Optional[str]) -> set:
        if dictionary_file:
            try:
                with open(dictionary_file, 'r') as f:
                    for word in f:
                        self.trie.insert(word.strip().upper())
            except FileNotFoundError:
                print("Dictionary file not found. Using minimal fallback dictionary.")
                for word in ["CAT", "DOG", "PYTHON"]:
                    self.trie.insert(word)
        return set()
    
    def add_player(self, player):
        """Add a player to the game"""
        self.players.append(player)
        player.tiles = self.tile_bag.draw_tiles(7)  # Each player starts with 7 tiles
    
    def is_valid_word(self, word: str) -> bool:
        return self.trie.is_word(word.upper())
    
    def calculate_word_score(self, word: str, start_pos: Tuple[int, int], direction: str) -> int:
        """Calculate score for placing a word"""
        score = 0
        word_multiplier = 1
        
        row, col = start_pos
        for i, letter in enumerate(word):
            letter_score = self.tile_bag.letter_values.get(letter, 0)
            
            if direction == 'horizontal':
                pos = (row, col + i)
            else:
                pos = (row + i, col)
            
            # Only apply premium if square is empty
            if self.board.board[pos[0]][pos[1]] == '':
                premium = self.board.premium_squares.get(pos)
                if premium == 'DLS':
                    letter_score *= 2
                elif premium == 'TLS':
                    letter_score *= 3
                elif premium == 'DWS':
                    word_multiplier *= 2
                elif premium == 'TWS':
                    word_multiplier *= 3
            
            score += letter_score
        
        return score * word_multiplier
    
    def get_valid_moves(self, player_tiles: List[str]) -> List[Dict]:
        """Get all valid moves for given tiles"""
        valid_moves = []
        
        # If board is empty, must place word through center
        if self.is_board_empty():
            center = (7, 7)
            # Try all possible words with player tiles
            for word in self._generate_possible_words(player_tiles):
                if self.is_valid_word(word):
                    # Try both directions through center
                    for direction in ['horizontal', 'vertical']:
                        start_pos = self._calculate_center_start(word, center, direction)
                        if self.board.is_valid_placement(word, start_pos, direction):
                            score = self.calculate_word_score(word, start_pos, direction)
                            valid_moves.append({
                                'word': word,
                                'position': start_pos,
                                'direction': direction,
                                'score': score
                            })
        else:
            # Find all anchor points (adjacent to existing tiles)
            anchor_points = self._find_anchor_points()
            for anchor in anchor_points:
                for word in self._generate_possible_words(player_tiles):
                    if self.is_valid_word(word):
                        for direction in ['horizontal', 'vertical']:
                            if self.board.is_valid_placement(word, anchor, direction):
                                score = self.calculate_word_score(word, anchor, direction)
                                valid_moves.append({
                                    'word': word,
                                    'position': anchor,
                                    'direction': direction,
                                    'score': score
                                })
        
        return valid_moves
    
    def _generate_possible_words(self, tiles: List[str]) -> List[str]:
        return self.trie.rack_words(tiles, max_words=300)

    
    def is_board_empty(self) -> bool:
        """Check if board is empty"""
        for row in self.board.board:
            for cell in row:
                if cell != '':
                    return False
        return True
    
    def _calculate_center_start(self, word: str, center: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate start position to place word through center"""
        row, col = center
        if direction == 'horizontal':
            return (row, col - len(word) // 2)
        else:
            return (row - len(word) // 2, col)
    
    def _find_anchor_points(self) -> List[Tuple[int, int]]:
        """Find positions adjacent to existing tiles"""
        anchors = []
        for row in range(self.board.size):
            for col in range(self.board.size):
                if self.board.board[row][col] == '':
                    # Check if adjacent to existing tile
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < self.board.size and 0 <= nc < self.board.size and
                            self.board.board[nr][nc] != ''):
                            anchors.append((row, col))
                            break
        return anchors
    
    def make_move(self, player_idx: int, move: Dict) -> bool:
        """Execute a move"""
        if player_idx >= len(self.players):
            return False
        
        player = self.players[player_idx]
        word = move['word']
        position = move['position']
        direction = move['direction']
        
        # Check if player has required tiles
        required_tiles = list(word)
        player_tiles = player.tiles.copy()
        
        for tile in required_tiles:
            if tile in player_tiles:
                player_tiles.remove(tile)
            else:
                return False  # Player doesn't have required tiles
        
        # Place word on board
        if self.board.place_word(word, position, direction):
            # Remove used tiles from player
            for tile in required_tiles:
                player.tiles.remove(tile)
            
            # Add new tiles
            new_tiles = self.tile_bag.draw_tiles(len(required_tiles))
            player.tiles.extend(new_tiles)
            
            # Update score
            player.score += move['score']
            
            self.turn_count += 1
            return True
        
        return False
    
    def get_game_state(self) -> Dict:
        """Get current game state"""
        return {
            'board': [row[:] for row in self.board.board],
            'players': [(p.score, len(p.tiles)) for p in self.players],
            'current_player': self.current_player,
            'tiles_remaining': self.tile_bag.remaining_count(),
            'turn_count': self.turn_count,
            'game_over': self.game_over
        }

class Player:
    """Base player class"""
    
    def __init__(self, name: str):
        self.name = name
        self.tiles = []
        self.score = 0
    
    def get_move(self, game: ScrabbleGame) -> Optional[Dict]:
        """Get player's move - to be implemented by subclasses"""
        raise NotImplementedError