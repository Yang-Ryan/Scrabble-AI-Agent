import os
import random
import sys
from typing import List, Dict, Optional
from pathlib import Path

# Import your existing modules
from scrabble_agent import AdaptiveScrabbleQLearner, GreedyAgent
from move_generator import MoveGenerator
from utils import (create_empty_board, create_tile_bag, draw_tiles, 
                  place_word_on_board, get_rack_after_move, create_game_state)

class SimpleScrabbleGame:
    
    def __init__(self, dictionary_path: str = 'dictionary.txt', model_path: str = None):
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
        self.model_path = "final_model.json"
        
        # Auto-load AI model
        if model_path:
            self.load_ai(model_path)
        else:
            # Try to find the most recent model
            self.auto_find_model()
    
    def auto_find_model(self):
        """Look for the single .json model file"""
        
        # Find all .json files
        json_files = list(Path('.').glob("*.json"))
        
        if len(json_files) == 1:
            # Perfect! Found exactly one .json file
            model_file = json_files[0]            
            if self.load_ai(str(model_file)):
                return
        elif len(json_files) > 1:
            # Multiple .json files - pick the most recent one
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model = json_files[0]
            
            if self.load_ai(str(latest_model)):
                return
        
        # No .json files found or loading failed
        print("⚠️ No .json model file found, using Greedy AI as opponent")
        self.set_greedy_ai()
    
    def clear_screen(self):
        """Clear terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_board(self):
        """Print the board with better formatting"""
        print("📋 Board:")
        print("   A B C D E F G H I J K L M N O")
        print("  " + "─" * 31)
        
        for i in range(15):
            row_str = f"{i+1:2d}│"
            for j in range(15):
                cell = self.board[i][j]
                if cell and cell != '':
                    row_str += f"{cell} "
                else:
                    # Show premium squares for empty cells
                    premium = self.get_premium_square(i, j)
                    if premium:
                        row_str += f"{premium} "
                    else:
                        row_str += "· "
            row_str += "│"
            print(row_str)
        print("  " + "─" * 31)
        
        # Legend
        print("Legend: DW=Double Word, TW=Triple Word, DL=Double Letter, TL=Triple Letter")
    
    def get_premium_square(self, row: int, col: int) -> str:
        """Get premium square indicator"""
        # Center star
        if row == 7 and col == 7:
            return "★"
        
        # Triple word scores
        triple_word = [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)]
        if (row, col) in triple_word:
            return "T"
        
        # Double word scores  
        double_word = [(1, 1), (2, 2), (3, 3), (4, 4), (1, 13), (2, 12), (3, 11), (4, 10),
                      (13, 1), (12, 2), (11, 3), (10, 4), (13, 13), (12, 12), (11, 11), (10, 10)]
        if (row, col) in double_word:
            return "D"
        
        return ""
    
    def print_status(self):
        """Print game status with emojis"""
        print(f"\n📝 Last move: {self.last_move}")
        print(f"🎲 Turn: {self.turn_number}")
        print(f"🏆 Scores - You: {self.human_score}, AI: {self.ai_score}")
        print(f"🤖 AI: {self.ai_name}")
        print(f"🎯 Tiles left: {len(self.tile_bag)}")
        print(f"🎪 Your rack: {' '.join(self.human_rack)}")
    
    def display_game(self):
        """Display complete game state"""
        self.clear_screen()
        print("🎮 SCRABBLE vs RL AGENT")
        print("=" * 50)
        self.print_board()
        self.print_status()
        print("=" * 50)
    
    def load_ai(self, model_path: str) -> bool:
        """Load AI model"""
        try:
            if not Path(model_path).exists():
                print(f"❌ File not found: {model_path}")
                return False
            
            self.ai_agent = AdaptiveScrabbleQLearner()
            self.ai_agent.load_model(model_path)
            
            # Get model info
            model_name = Path(model_path).stem
            episodes = getattr(self.ai_agent, 'training_episodes', 'Unknown')
            
            self.ai_name = f"RL Agent ({model_name}, {episodes} episodes)"
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def set_greedy_ai(self):
        """Use greedy AI as fallback"""
        self.ai_agent = GreedyAgent()
        self.ai_name = "Greedy AI (Fallback)"
        print("✅ Using Greedy AI")
    
    def get_human_moves(self) -> List[Dict]:
        """Get valid moves for human"""
        return self.move_generator.get_valid_moves(self.board, self.human_rack)
    
    def show_moves(self, moves: List[Dict], limit: int = 15):
        """Show available moves with better formatting"""
        if not moves:
            print("❌ No valid moves!")
            return
        
        print(f"\n📋 Your moves (showing top {min(len(moves), limit)} by score):")
        
        # Sort by score for better display
        sorted_moves = sorted(moves, key=lambda m: m['score'], reverse=True)
        
        for i, move in enumerate(sorted_moves[:limit]):
            row, col = move['position']
            pos = f"({row+1:2d},{chr(65+col)})"
            direction = "→" if move.get('direction') == 'horizontal' else "↓"
            print(f"{i+1:2d}. {move['word']:10s} - {move['score']:3d} pts - {pos} {direction}")
        
        if len(moves) > limit:
            print(f"... and {len(moves) - limit} more moves available")
    
    def human_turn(self):
        """Handle human turn with improved interface"""
        moves = self.get_human_moves()
        
        if not moves:
            # No moves available
            print("❌ No valid moves available!")
            print("\n🔄 Options:")
            print("1. Exchange tiles (need ≥7 tiles in bag)")
            print("2. Pass turn")
            
            while True:
                choice = input("\n🎮 Choose option (1/2): ").strip()
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
                    print("Please enter 1 or 2")
        else:
            # Show moves and get choice
            self.show_moves(moves)
            
            print(f"\n🎮 Commands:")
            print(f"  • Enter move number (1-{min(len(moves), 15)})")
            print(f"  • 'more' - show all moves")
            print(f"  • 'pass' - skip turn")
            print(f"  • 'quit' - end game")
            
            while True:
                choice = input("\n🎯 Your choice: ").strip().lower()
                
                if choice == 'quit':
                    self.game_over = True
                    return
                elif choice == 'pass':
                    self.last_move = "You passed"
                    self.consecutive_passes += 1
                    return
                elif choice == 'more':
                    self.show_moves(moves, len(moves))
                    continue
                
                try:
                    move_num = int(choice)
                    # Sort moves by score to match display
                    sorted_moves = sorted(moves, key=lambda m: m['score'], reverse=True)
                    if 1 <= move_num <= len(sorted_moves):
                        self.play_move(sorted_moves[move_num - 1], is_human=True)
                        self.consecutive_passes = 0
                        return
                    else:
                        print(f"Please enter 1-{len(sorted_moves)}")
                except ValueError:
                    print("Please enter a valid number")
    
    def exchange_tiles(self):
        """Exchange tiles with improved interface"""
        print(f"\n🔄 Current rack: {' '.join(f'{i+1}:{tile}' for i, tile in enumerate(self.human_rack))}")
        print("Enter positions to exchange (e.g., '1 3 5') or 'all':")
        
        while True:
            choice = input("🔄 Exchange: ").strip().lower()
            
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
                        print("❌ Invalid positions")
                except ValueError:
                    print("❌ Enter numbers like '1 3 5'")
        
        # Exchange tiles
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
        """Execute a move with better feedback"""
        # Place word on board
        self.board = place_word_on_board(self.board, move['word'], move['positions'])
        
        # Update score
        if is_human:
            self.human_score += move['score']
            player = "You"
            emoji = "🎯"
        else:
            self.ai_score += move['score']
            player = "AI"
            emoji = "🤖"
        
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
        
        # Special move notifications
        bonus = ""
        if move['score'] >= 50:
            bonus = " 🔥 HIGH SCORE!"
        elif len(move['word']) >= 7:
            bonus = " ⭐ BINGO!"
        
        self.last_move = f"{emoji} {player} played '{move['word']}' for {move['score']} points{bonus}"
    
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
        """Show game result with stats"""
        print("\n🏆 GAME OVER!")
        print("=" * 40)
        print(f"📊 Final Scores:")
        print(f"   You: {self.human_score}")
        print(f"   AI:  {self.ai_score}")
        
        diff = abs(self.human_score - self.ai_score)
        if self.human_score > self.ai_score:
            print(f"\n🎉 Congratulations! You won by {diff} points!")
        elif self.ai_score > self.human_score:
            print(f"\n🤖 AI won by {diff} points! Good game!")
        else:
            print(f"\n🤝 Amazing! It's a perfect tie!")
        
        print(f"\n📈 Game Stats:")
        print(f"   Game lasted: {self.turn_number} turns")
        print(f"   AI opponent: {self.ai_name}")
        print(f"   Your avg per turn: {self.human_score/self.turn_number:.1f}")
        print(f"   AI avg per turn: {self.ai_score/self.turn_number:.1f}")
    
    def play(self):
        """Main game loop"""
        print("==============================\n🎮 Starting Scrabble vs RL Agent!")
        input("Press Enter to begin...")
        
        while not self.game_over:
            # Display game
            self.display_game()
            
            # Check game over
            if self.is_game_over():
                break
            
            # Human turn
            print("\n🎯 YOUR TURN!")
            self.human_turn()
            
            if self.game_over:
                break
            
            # AI turn
            print("\n🤖 AI is thinking...")
            input("Press Enter to see AI move...")
            self.ai_turn()
            
            # Next turn
            self.turn_number += 1
        
        # Show final result
        self.display_game()
        self.show_final_result()

def main():
    """Start the game with optional model path"""
    
    # Check for command line model path
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"📦 Using specified model: {model_path}")
    
    # Create and start game
    game = SimpleScrabbleGame(model_path=model_path)
    game.play()

if __name__ == "__main__":
    main()