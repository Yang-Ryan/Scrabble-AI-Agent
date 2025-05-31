"""
Simple Terminal Scrabble Game Interface
Human vs AI with proper Scrabble rules
"""

import os
import random
from typing import List, Dict, Optional
from pathlib import Path

# Import your existing modules
from scrabble_agent import AdaptiveScrabbleQLearner, GreedyAgent
from move_generator import MoveGenerator
from utils import (create_empty_board, create_tile_bag, draw_tiles, 
                  place_word_on_board, get_rack_after_move, create_game_state)

class SimpleScrabbleGame:
    """
    Clean and simple terminal Scrabble game
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        # Initialize game components
        self.board = create_empty_board()
        self.tile_bag = create_tile_bag()
        self.move_generator = MoveGenerator(dictionary_path)
        
        # Player racks and scores
        self.human_rack = draw_tiles(self.tile_bag, 7)
        self.ai_rack = draw_tiles(self.tile_bag, 7)
        self.human_score = 0
        self.ai_score = 0
        
        # Game state
        self.turn_number = 1
        self.last_move = "Game started"
        self.consecutive_passes = 0
        self.game_over = False
        
        # AI
        self.ai_agent = None
        self.ai_name = "No AI"
    
    def clear_screen(self):
        """Clear terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_board(self):
        """Print the board"""
        print("Board:")
        print("  A B C D E F G H I J K L M N O")
        print("  " + "-" * 29)
        
        for i in range(15):
            row_str = f"{i+1:2d}|"
            for j in range(15):
                cell = self.board[i][j]
                if cell and cell != '':
                    row_str += f"{cell} "
                else:
                    row_str += "  "
            row_str += "|"
            print(row_str)
        print("  " + "-" * 29)
    
    def print_status(self):
        """Print game status"""
        print(f"\nLast move: {self.last_move}")
        print(f"Turn: {self.turn_number}")
        print(f"Scores - You: {self.human_score}, AI: {self.ai_score}")
        print(f"AI: {self.ai_name}")
        print(f"Tiles left: {len(self.tile_bag)}")
        print(f"Your rack: {' '.join(self.human_rack)}")
    
    def display_game(self):
        """Display complete game state"""
        self.clear_screen()
        print("🎮 SCRABBLE GAME")
        print("=" * 40)
        self.print_board()
        self.print_status()
        print("=" * 40)
    
    def load_ai(self, model_path: str) -> bool:
        """Load AI model"""
        try:
            if not Path(model_path).exists():
                print(f"❌ File not found: {model_path}")
                return False
            
            self.ai_agent = AdaptiveScrabbleQLearner()
            self.ai_agent.load_model(model_path)
            self.ai_name = f"RL Model ({Path(model_path).stem})"
            print(f"✅ Loaded: {self.ai_name}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def set_greedy_ai(self):
        """Use greedy AI"""
        self.ai_agent = GreedyAgent()
        self.ai_name = "Greedy AI"
        print("✅ Using Greedy AI")
    
    def get_human_moves(self) -> List[Dict]:
        """Get valid moves for human"""
        return self.move_generator.get_valid_moves(self.board, self.human_rack)
    
    def show_moves(self, moves: List[Dict], limit: int = 10):
        """Show available moves"""
        if not moves:
            print("❌ No valid moves!")
            return
        
        print(f"\n📋 Your moves (showing {min(len(moves), limit)}):")
        for i, move in enumerate(moves[:limit]):
            pos = f"({move['position'][0]+1},{chr(65+move['position'][1])})"
            print(f"{i+1:2d}. {move['word']:8s} - {move['score']:3d} pts - {pos}")
        
        if len(moves) > limit:
            print(f"... and {len(moves) - limit} more")
    
    def human_turn(self):
        """Handle human turn"""
        moves = self.get_human_moves()
        
        if not moves:
            # No moves available
            print("❌ No valid moves available!")
            print("Options:")
            print("1. Exchange tiles (need ≥7 tiles in bag)")
            print("2. Pass turn")
            
            while True:
                choice = input("Choose (1/2): ").strip()
                if choice == '1':
                    if len(self.tile_bag) >= 7:
                        self.exchange_tiles()
                        self.consecutive_passes = 0
                        return
                    else:
                        print("❌ Not enough tiles to exchange")
                elif choice == '2':
                    self.last_move = "You passed"
                    self.consecutive_passes += 1
                    return
                else:
                    print("Enter 1 or 2")
        else:
            # Show moves and get choice
            self.show_moves(moves)
            
            while True:
                choice = input("\n🎮 Move number (or 'pass'/'quit'): ").strip().lower()
                
                if choice == 'quit':
                    self.game_over = True
                    return
                elif choice == 'pass':
                    self.last_move = "You passed"
                    self.consecutive_passes += 1
                    return
                
                try:
                    move_num = int(choice)
                    if 1 <= move_num <= len(moves):
                        self.play_move(moves[move_num - 1], is_human=True)
                        self.consecutive_passes = 0
                        return
                    else:
                        print(f"Enter 1-{len(moves)}")
                except ValueError:
                    print("Enter a number")
    
    def exchange_tiles(self):
        """Exchange tiles"""
        print(f"Current rack: {' '.join(self.human_rack)}")
        print("Enter positions to exchange (e.g. '1 3 5') or 'all':")
        
        while True:
            choice = input("Exchange: ").strip().lower()
            
            if choice == 'all':
                tiles_to_exchange = self.human_rack.copy()
                break
            else:
                try:
                    positions = [int(x) - 1 for x in choice.split()]
                    if all(0 <= pos < len(self.human_rack) for pos in positions):
                        tiles_to_exchange = [self.human_rack[pos] for pos in positions]
                        break
                    else:
                        print("Invalid positions")
                except ValueError:
                    print("Enter numbers like '1 3 5'")
        
        # Exchange
        for tile in tiles_to_exchange:
            self.human_rack.remove(tile)
        
        self.tile_bag.extend(tiles_to_exchange)
        random.shuffle(self.tile_bag)
        
        new_tiles = draw_tiles(self.tile_bag, len(tiles_to_exchange))
        self.human_rack.extend(new_tiles)
        
        self.last_move = f"You exchanged {len(tiles_to_exchange)} tiles"
        print(f"✅ Exchanged {len(tiles_to_exchange)} tiles")
    
    def ai_turn(self):
        """Handle AI turn"""
        if not self.ai_agent:
            print("❌ No AI loaded!")
            return
        
        # Get AI moves
        ai_moves = self.move_generator.get_valid_moves(self.board, self.ai_rack)
        
        if not ai_moves:
            # AI has no moves - decide to exchange or pass
            if len(self.tile_bag) >= 7 and self.should_ai_exchange():
                self.ai_exchange()
                self.consecutive_passes = 0
            else:
                self.last_move = "AI passed"
                self.consecutive_passes += 1
        else:
            # AI chooses move
            ai_state = create_game_state(
                self.board, self.ai_rack, [], self.ai_score, self.human_score,
                len(self.tile_bag), self.turn_number
            )
            
            chosen_move = self.ai_agent.choose_move(ai_state, ai_moves, training=False)
            
            if chosen_move:
                self.play_move(chosen_move, is_human=False)
                self.consecutive_passes = 0
            else:
                self.last_move = "AI passed"
                self.consecutive_passes += 1
    
    def should_ai_exchange(self) -> bool:
        """Simple AI exchange logic"""
        # Count vowels and duplicates
        vowels = sum(1 for tile in self.ai_rack if tile in 'AEIOU')
        vowel_ratio = vowels / len(self.ai_rack)
        
        # Count duplicates
        tile_counts = {}
        for tile in self.ai_rack:
            tile_counts[tile] = tile_counts.get(tile, 0) + 1
        duplicates = sum(1 for count in tile_counts.values() if count > 1)
        
        # Exchange if bad rack
        return duplicates >= 3 or vowel_ratio < 0.15 or vowel_ratio > 0.8
    
    def ai_exchange(self):
        """AI exchanges tiles"""
        # Exchange 3-5 random tiles
        num_exchange = min(random.randint(3, 5), len(self.ai_rack))
        tiles_to_exchange = random.sample(self.ai_rack, num_exchange)
        
        for tile in tiles_to_exchange:
            self.ai_rack.remove(tile)
        
        self.tile_bag.extend(tiles_to_exchange)
        random.shuffle(self.tile_bag)
        
        new_tiles = draw_tiles(self.tile_bag, len(tiles_to_exchange))
        self.ai_rack.extend(new_tiles)
        
        self.last_move = f"AI exchanged {len(tiles_to_exchange)} tiles"
    
    def play_move(self, move: Dict, is_human: bool):
        """Execute a move"""
        # Place word on board
        self.board = place_word_on_board(self.board, move['word'], move['positions'])
        
        # Update score
        if is_human:
            self.human_score += move['score']
            player = "You"
        else:
            self.ai_score += move['score']
            player = "AI"
        
        # Update rack
        if is_human:
            tiles_drawn = draw_tiles(self.tile_bag, len(move['tiles_used']))
            self.human_rack = get_rack_after_move(
                self.human_rack, move['tiles_used'], tiles_drawn
            )
        else:
            tiles_drawn = draw_tiles(self.tile_bag, len(move['tiles_used']))
            self.ai_rack = get_rack_after_move(
                self.ai_rack, move['tiles_used'], tiles_drawn
            )
        
        self.last_move = f"{player} played '{move['word']}' for {move['score']} points"
    
    def is_game_over(self) -> bool:
        """Check if game should end"""
        # Two consecutive passes
        if self.consecutive_passes >= 2:
            return True
        
        # No tiles left and someone has empty rack
        if len(self.tile_bag) == 0:
            if len(self.human_rack) == 0 or len(self.ai_rack) == 0:
                return True
        
        # Too many turns (safety)
        if self.turn_number > 100:
            return True
        
        return False
    
    def show_final_result(self):
        """Show game result"""
        print("\n🏆 GAME OVER!")
        print("=" * 25)
        print(f"Final Scores:")
        print(f"You: {self.human_score}")
        print(f"AI:  {self.ai_score}")
        
        diff = abs(self.human_score - self.ai_score)
        if self.human_score > self.ai_score:
            print(f"🎉 You won by {diff} points!")
        elif self.ai_score > self.human_score:
            print(f"🤖 AI won by {diff} points!")
        else:
            print("🤝 It's a tie!")
        
        print(f"Game lasted {self.turn_number} turns")
    
    def play(self):
        """Main game loop"""
        print("🎮 Starting Scrabble Game!")
        
        while not self.game_over:
            # Display game
            self.display_game()
            
            # Check game over
            if self.is_game_over():
                break
            
            # Human turn
            print("\n🎯 Your turn!")
            self.human_turn()
            
            if self.game_over:
                break
            
            # AI turn
            print("\n🤖 AI's turn...")
            input("Press Enter for AI move...")
            self.ai_turn()
            
            # Next turn
            self.turn_number += 1
        
        # Show final result
        self.display_game()
        self.show_final_result()

def main():
    """Start the game"""
    game = SimpleScrabbleGame()
    
    print("🎮 SCRABBLE SETUP")
    print("=" * 20)
    print("Choose AI opponent:")
    print("1. Greedy AI")
    print("2. Load RL model")
    
    while True:
        choice = input("Choice (1/2): ").strip()
        if choice == '1':
            game.set_greedy_ai()
            break
        elif choice == '2':
            model_path = input("Model path: ").strip()
            if game.load_ai(model_path):
                break
        else:
            print("Enter 1 or 2")
    
    input("\nPress Enter to start...")
    game.play()

if __name__ == "__main__":
    main()