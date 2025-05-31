"""
Human vs AI Scrabble Game Interface
Simple GUI for playing against trained AI agents
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
from typing import List, Dict, Tuple, Optional
import threading
import time

# Import your existing components
from scrabble_agent import AdaptiveScrabbleQLearner
from move_generator import MoveGenerator
from utils import (create_empty_board, create_tile_bag, draw_tiles, 
                  place_word_on_board, get_rack_after_move, create_game_state,
                  TILE_VALUES, board_to_string)

class ScrabbleGameGUI:
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        self.root = tk.Tk()
        self.root.title("🎮 Human vs AI Scrabble")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Game state
        self.board = create_empty_board()
        self.tile_bag = create_tile_bag()
        self.human_rack = draw_tiles(self.tile_bag, 7)
        self.ai_rack = draw_tiles(self.tile_bag, 7)
        self.human_score = 0
        self.ai_score = 0
        self.game_over = False
        self.current_turn = 'human'
        
        # AI and move generator
        self.ai_agent = None
        self.move_generator = MoveGenerator(dictionary_path)
        
        # Selected tiles for human move
        self.selected_tiles = []
        self.selected_word = ""
        self.selected_positions = []
        
        # GUI elements
        self.board_buttons = []
        self.rack_buttons = []
        
        self.setup_gui()
        self.update_display()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        self.setup_control_panel(main_frame)
        
        # Game area
        game_frame = tk.Frame(main_frame, bg='#2c3e50')
        game_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side - Board
        self.setup_board(game_frame)
        
        # Right side - Info panel
        self.setup_info_panel(game_frame)
        
        # Bottom - Human rack and controls
        self.setup_human_controls(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup top control panel"""
        control_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load AI model button
        tk.Button(control_frame, text="🤖 Load AI Model", 
                 command=self.load_ai_model, bg='#3498db', fg='white',
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # New game button
        tk.Button(control_frame, text="🎮 New Game", 
                 command=self.new_game, bg='#27ae60', fg='white',
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # AI difficulty (future feature)
        tk.Label(control_frame, text="AI Status:", bg='#34495e', fg='white',
                font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        
        self.ai_status_label = tk.Label(control_frame, text="No AI Loaded", 
                                       bg='#34495e', fg='#e74c3c',
                                       font=('Arial', 10, 'bold'))
        self.ai_status_label.pack(side=tk.LEFT)
        
        # Game status
        self.turn_label = tk.Label(control_frame, text="Your Turn", 
                                  bg='#34495e', fg='#f39c12',
                                  font=('Arial', 12, 'bold'))
        self.turn_label.pack(side=tk.RIGHT, padx=10)
        
    def setup_board(self, parent):
        """Setup the Scrabble board"""
        board_frame = tk.Frame(parent, bg='#2c3e50')
        board_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(board_frame, text="🎯 SCRABBLE BOARD", bg='#2c3e50', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Board grid
        board_grid = tk.Frame(board_frame, bg='#34495e', relief=tk.RAISED, bd=3)
        board_grid.pack()
        
        self.board_buttons = []
        for row in range(15):
            button_row = []
            for col in range(15):
                # Special square colors
                bg_color = self.get_square_color(row, col)
                
                btn = tk.Button(board_grid, text='', width=3, height=1,
                               bg=bg_color, font=('Arial', 10, 'bold'),
                               command=lambda r=row, c=col: self.board_click(r, c))
                btn.grid(row=row, column=col, padx=1, pady=1)
                button_row.append(btn)
            self.board_buttons.append(button_row)
    
    def get_square_color(self, row: int, col: int) -> str:
        """Get the color for a board square based on premium squares"""
        # Center square
        if row == 7 and col == 7:
            return '#f39c12'  # Orange for center
        
        # Triple Word Score
        if (row, col) in [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), 
                         (14, 0), (14, 7), (14, 14)]:
            return '#e74c3c'  # Red
        
        # Double Word Score  
        if (row, col) in [(1, 1), (2, 2), (3, 3), (4, 4), (10, 10), 
                         (11, 11), (12, 12), (13, 13)] or \
           (row, col) in [(1, 13), (2, 12), (3, 11), (4, 10), (10, 4), 
                         (11, 3), (12, 2), (13, 1)]:
            return '#e67e22'  # Dark orange
        
        # Triple Letter Score
        if (row, col) in [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
                         (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)]:
            return '#3498db'  # Blue
        
        # Double Letter Score
        if (row, col) in [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), 
                         (3, 14), (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), 
                         (7, 11), (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), 
                         (11, 7), (11, 14), (12, 6), (12, 8), (14, 3), (14, 11)]:
            return '#95a5a6'  # Light gray
        
        return '#ecf0f1'  # Default light gray
    
    def setup_info_panel(self, parent):
        """Setup the right info panel"""
        info_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=2)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        
        tk.Label(info_frame, text="📊 GAME INFO", bg='#34495e', fg='white',
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Scores
        score_frame = tk.Frame(info_frame, bg='#34495e')
        score_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(score_frame, text="SCORES", bg='#34495e', fg='white',
                font=('Arial', 11, 'bold')).pack()
        
        self.human_score_label = tk.Label(score_frame, text="You: 0", 
                                         bg='#34495e', fg='#2ecc71',
                                         font=('Arial', 14, 'bold'))
        self.human_score_label.pack(pady=2)
        
        self.ai_score_label = tk.Label(score_frame, text="AI: 0", 
                                      bg='#34495e', fg='#e74c3c',
                                      font=('Arial', 14, 'bold'))
        self.ai_score_label.pack(pady=2)
        
        # Game info
        info_text_frame = tk.Frame(info_frame, bg='#34495e')
        info_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(info_text_frame, text="GAME LOG", bg='#34495e', fg='white',
                font=('Arial', 11, 'bold')).pack()
        
        # Scrollable text area for game log
        log_frame = tk.Frame(info_text_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.game_log = tk.Text(log_frame, width=25, height=15, 
                               yscrollcommand=scrollbar.set,
                               font=('Courier', 9), bg='#2c3e50', fg='white')
        self.game_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.game_log.yview)
        
        # AI rack display
        ai_rack_frame = tk.Frame(info_frame, bg='#34495e')
        ai_rack_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(ai_rack_frame, text="AI RACK", bg='#34495e', fg='white',
                font=('Arial', 11, 'bold')).pack()
        
        self.ai_rack_label = tk.Label(ai_rack_frame, text="? ? ? ? ? ? ?", 
                                     bg='#34495e', fg='#95a5a6',
                                     font=('Arial', 12, 'bold'))
        self.ai_rack_label.pack(pady=5)
        
        # Tiles remaining
        self.tiles_remaining_label = tk.Label(info_frame, text="Tiles: 98", 
                                             bg='#34495e', fg='white',
                                             font=('Arial', 10))
        self.tiles_remaining_label.pack(pady=5)
    
    def setup_human_controls(self, parent):
        """Setup human player controls"""
        controls_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=2)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(controls_frame, text="🎮 YOUR CONTROLS", bg='#34495e', fg='white',
                font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Human rack
        rack_frame = tk.Frame(controls_frame, bg='#34495e')
        rack_frame.pack(pady=10)
        
        tk.Label(rack_frame, text="Your Tiles:", bg='#34495e', fg='white',
                font=('Arial', 11, 'bold')).pack()
        
        self.rack_buttons_frame = tk.Frame(rack_frame, bg='#34495e')
        self.rack_buttons_frame.pack(pady=5)
        
        # Move input
        move_frame = tk.Frame(controls_frame, bg='#34495e')
        move_frame.pack(pady=10)
        
        tk.Label(move_frame, text="Word:", bg='#34495e', fg='white',
                font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.word_entry = tk.Entry(move_frame, font=('Arial', 12), width=15)
        self.word_entry.pack(side=tk.LEFT, padx=5)
        self.word_entry.bind('<Return>', self.on_word_entry)
        
        tk.Button(move_frame, text="✨ Get Suggestions", 
                 command=self.get_move_suggestions, bg='#9b59b6', fg='white',
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = tk.Frame(controls_frame, bg='#34495e')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="✅ Play Move", 
                 command=self.play_human_move, bg='#27ae60', fg='white',
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🔄 Pass Turn", 
                 command=self.pass_turn, bg='#f39c12', fg='white',
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🔀 Exchange Tiles", 
                 command=self.exchange_tiles, bg='#95a5a6', fg='white',
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(controls_frame, text="Click tiles to select, then enter word", 
                                    bg='#34495e', fg='#bdc3c7',
                                    font=('Arial', 10))
        self.status_label.pack(pady=5)
    
    def board_click(self, row: int, col: int):
        """Handle board square click"""
        if self.current_turn != 'human' or self.game_over:
            return
        
        # For now, just show coordinates
        self.log_message(f"Clicked board position ({row}, {col})")
    
    def create_rack_buttons(self):
        """Create buttons for human rack tiles"""
        # Clear existing buttons
        for widget in self.rack_buttons_frame.winfo_children():
            widget.destroy()
        
        self.rack_buttons = []
        for i, tile in enumerate(self.human_rack):
            btn = tk.Button(self.rack_buttons_frame, text=tile, width=4, height=2,
                           bg='#ecf0f1', font=('Arial', 12, 'bold'),
                           command=lambda idx=i: self.rack_tile_click(idx))
            btn.pack(side=tk.LEFT, padx=2)
            self.rack_buttons.append(btn)
    
    def rack_tile_click(self, index: int):
        """Handle rack tile click"""
        if self.current_turn != 'human' or self.game_over:
            return
        
        tile = self.human_rack[index]
        
        if index in self.selected_tiles:
            # Deselect tile
            self.selected_tiles.remove(index)
            self.rack_buttons[index].config(bg='#ecf0f1')
        else:
            # Select tile
            self.selected_tiles.append(index)
            self.rack_buttons[index].config(bg='#f39c12')
        
        # Update selected word display
        selected_letters = [self.human_rack[i] for i in sorted(self.selected_tiles)]
        self.selected_word = ''.join(selected_letters)
        self.word_entry.delete(0, tk.END)
        self.word_entry.insert(0, self.selected_word)
        
        self.update_status(f"Selected: {self.selected_word}")
    
    def on_word_entry(self, event=None):
        """Handle word entry"""
        word = self.word_entry.get().upper().strip()
        if word:
            self.selected_word = word
            self.update_status(f"Word entered: {word}")
    
    def get_move_suggestions(self):
        """Get AI suggestions for human player"""
        if self.current_turn != 'human' or self.game_over:
            return
        
        try:
            # Get valid moves for human rack
            valid_moves = self.move_generator.get_valid_moves(self.board, self.human_rack)
            
            if not valid_moves:
                self.log_message("No valid moves found!")
                return
            
            # Show top 5 suggestions
            suggestions = sorted(valid_moves, key=lambda m: m['score'], reverse=True)[:5]
            
            self.log_message("🔮 MOVE SUGGESTIONS:")
            for i, move in enumerate(suggestions, 1):
                self.log_message(f"{i}. {move['word']} - {move['score']} pts")
            
        except Exception as e:
            self.log_message(f"Error getting suggestions: {e}")
    
    def play_human_move(self):
        """Play human move"""
        if self.current_turn != 'human' or self.game_over:
            return
        
        word = self.word_entry.get().upper().strip()
        if not word:
            messagebox.showwarning("Invalid Move", "Please enter a word!")
            return
        
        # Simple validation - try to find this word in valid moves
        valid_moves = self.move_generator.get_valid_moves(self.board, self.human_rack)
        matching_moves = [m for m in valid_moves if m['word'] == word]
        
        if not matching_moves:
            messagebox.showwarning("Invalid Move", f"'{word}' is not a valid move!")
            return
        
        # Use the highest scoring version of this word
        move = max(matching_moves, key=lambda m: m['score'])
        
        # Execute move
        self.execute_human_move(move)
    
    def execute_human_move(self, move: Dict):
        """Execute a validated human move"""
        # Update board
        self.board = place_word_on_board(self.board, move['word'], move['positions'])
        
        # Update score
        self.human_score += move['score']
        
        # Update rack
        tiles_drawn = draw_tiles(self.tile_bag, len(move['tiles_used']))
        self.human_rack = get_rack_after_move(self.human_rack, move['tiles_used'], tiles_drawn)
        
        # Log move
        self.log_message(f"🎯 You played: {move['word']} ({move['score']} pts)")
        
        # Clear selection
        self.selected_tiles = []
        self.selected_word = ""
        self.word_entry.delete(0, tk.END)
        
        # Switch turns
        self.current_turn = 'ai'
        self.update_display()
        
        # AI move after short delay
        self.root.after(1000, self.ai_move)
    
    def pass_turn(self):
        """Pass turn"""
        if self.current_turn != 'human' or self.game_over:
            return
        
        self.log_message("🔄 You passed your turn")
        self.current_turn = 'ai'
        self.update_display()
        self.root.after(1000, self.ai_move)
    
    def exchange_tiles(self):
        """Exchange selected tiles"""
        if self.current_turn != 'human' or self.game_over:
            return
        
        if not self.selected_tiles:
            messagebox.showwarning("No Selection", "Please select tiles to exchange!")
            return
        
        # Exchange tiles
        exchanged_tiles = [self.human_rack[i] for i in sorted(self.selected_tiles, reverse=True)]
        
        # Remove from rack
        for i in sorted(self.selected_tiles, reverse=True):
            self.human_rack.pop(i)
        
        # Add back to bag and shuffle
        self.tile_bag.extend(exchanged_tiles)
        random.shuffle(self.tile_bag)
        
        # Draw new tiles
        new_tiles = draw_tiles(self.tile_bag, len(exchanged_tiles))
        self.human_rack.extend(new_tiles)
        
        self.log_message(f"🔀 Exchanged {len(exchanged_tiles)} tiles")
        
        # Clear selection
        self.selected_tiles = []
        self.word_entry.delete(0, tk.END)
        
        # Switch turns
        self.current_turn = 'ai'
        self.update_display()
        self.root.after(1000, self.ai_move)
    
    def ai_move(self):
        """Execute AI move"""
        if self.current_turn != 'ai' or self.game_over or not self.ai_agent:
            return
        
        try:
            # Create game state
            game_state = create_game_state(
                self.board, self.ai_rack, [], self.ai_score, self.human_score,
                len(self.tile_bag), 0
            )
            
            # Get valid moves
            valid_moves = self.move_generator.get_valid_moves(self.board, self.ai_rack)
            
            if not valid_moves:
                self.log_message("🤖 AI passes (no valid moves)")
                self.current_turn = 'human'
                self.update_display()
                return
            
            # AI chooses move
            chosen_move = self.ai_agent.choose_move(game_state, valid_moves, training=False)
            
            if not chosen_move:
                self.log_message("🤖 AI passes")
                self.current_turn = 'human'
                self.update_display()
                return
            
            # Execute AI move
            self.board = place_word_on_board(self.board, chosen_move['word'], chosen_move['positions'])
            self.ai_score += chosen_move['score']
            
            # Update AI rack
            tiles_drawn = draw_tiles(self.tile_bag, len(chosen_move['tiles_used']))
            self.ai_rack = get_rack_after_move(self.ai_rack, chosen_move['tiles_used'], tiles_drawn)
            
            # Log move
            self.log_message(f"🤖 AI played: {chosen_move['word']} ({chosen_move['score']} pts)")
            
            # Switch turns
            self.current_turn = 'human'
            self.update_display()
            
        except Exception as e:
            self.log_message(f"❌ AI error: {e}")
            self.current_turn = 'human'
            self.update_display()
    
    def load_ai_model(self, file_path=None):
        """Load AI model from file"""
        if not file_path:
            file_path = filedialog.askopenfilename(
                title="Select AI Model",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
        
        if file_path:
            try:
                self.ai_agent = AdaptiveScrabbleQLearner()
                self.ai_agent.load_model(file_path)
                
                self.ai_status_label.config(text="AI Loaded ✅", fg='#27ae60')
                self.log_message(f"🤖 Loaded AI model: {file_path.split('/')[-1]}")
                self.log_message(f"   Training episodes: {self.ai_agent.training_episodes}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load AI model: {e}")
                self.ai_status_label.config(text="Load Failed ❌", fg='#e74c3c')

    
    
    def new_game(self):
        """Start a new game"""
        # Reset game state
        self.board = create_empty_board()
        self.tile_bag = create_tile_bag()
        self.human_rack = draw_tiles(self.tile_bag, 7)
        self.ai_rack = draw_tiles(self.tile_bag, 7)
        self.human_score = 0
        self.ai_score = 0
        self.game_over = False
        self.current_turn = 'human'
        
        # Clear selections
        self.selected_tiles = []
        self.selected_word = ""
        
        # Clear log
        self.game_log.delete(1.0, tk.END)
        
        self.log_message("🎮 New game started!")
        self.log_message("Make your first move!")
        
        self.update_display()
    
    def update_display(self):
        """Update all GUI elements"""
        # Update board
        for row in range(15):
            for col in range(15):
                tile = self.board[row][col]
                if tile:
                    self.board_buttons[row][col].config(text=tile, bg='#f8c471', fg='black')
                else:
                    bg_color = self.get_square_color(row, col)
                    self.board_buttons[row][col].config(text='', bg=bg_color)
        
        # Update scores
        self.human_score_label.config(text=f"You: {self.human_score}")
        self.ai_score_label.config(text=f"AI: {self.ai_score}")
        
        # Update turn indicator
        if self.current_turn == 'human':
            self.turn_label.config(text="🎯 Your Turn", fg='#2ecc71')
        else:
            self.turn_label.config(text="🤖 AI Turn", fg='#e74c3c')
        
        # Update human rack
        self.create_rack_buttons()
        
        # Update tiles remaining
        self.tiles_remaining_label.config(text=f"Tiles: {len(self.tile_bag)}")
        
        # Show AI rack (hidden)
        ai_rack_display = " ".join(["?" for _ in self.ai_rack])
        self.ai_rack_label.config(text=ai_rack_display)
    
    def update_status(self, message: str):
        """Update status message"""
        self.status_label.config(text=message)
    
    def log_message(self, message: str):
        """Add message to game log"""
        self.game_log.insert(tk.END, message + "\n")
        self.game_log.see(tk.END)
    
    def run(self):
        """Start the GUI"""
        self.log_message("🎮 Welcome to Human vs AI Scrabble!")
        self.log_message("📁 Load an AI model to start playing")
        self.log_message("🎯 Click 'New Game' when ready")
        self.root.mainloop()

def main():
    """Launch the Human vs AI game"""
    print("🎮 Starting Human vs AI Scrabble Game...")
    game = ScrabbleGameGUI()
    game.run()

if __name__ == "__main__":
    main()