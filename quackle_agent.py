from datetime import datetime
from quackle_project.bindings.python3 import quackle_agent as quackle_py
import os
from move_generator import MoveGenerator
from typing import List, Dict, Optional

column_mapping = {
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
    "6": "F", "7": "G", "8": "H", "9": "I", "10": "J",
    "11": "K", "12": "L", "13": "M", "14": "N", "15": "O",
} #1-base mapping


class QuackleAgent:

    def __init__(self):
        now = datetime.now()
        self.move_generator = MoveGenerator()

        #gcg file in python 3 direction
        self.gcg_name = now.strftime("%Y%m%d_%H%M%S%f")[:-3] + ".gcg"
        
        # record both total score
        self.us_total_score = 0
        self.quackle_total_score = 0

        # init gcg 
        try:
            with open(self.gcg_name, "w", encoding="utf-8") as f:
                f.write("#player1 us us\n")
                f.write("#player2 quackle quackle\n")
                f.write("#rack2 \n") # wait for quackle 抽牌
            print(f"檔案初始化並儲存成功！路徑：{os.path.abspath(self.gcg_name)}")
        except PermissionError:
            print("權限錯誤！無法寫入檔案，請檢查目錄權限或換個路徑！")
        except Exception as e:
            print(f"初始化檔案時出錯：{e}")


    # recrod ( without thinking )
    def to_quackle_input(self, player: str, move: Dict):
        """
           player: "us" or "quackle"
           move:  {
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

            write a line to gcg "
                >[player]: [rack before move]
                [move start point] [tiles used] 
                [score increment] [total score]"
        """


        # rack before move
        rack_before_move = move["tiles_used"] + move["remaining_rack"]
        rack_before_move_str = "".join(rack_before_move).replace(".", "")

        # move start point(from 0-base to 1-base)
        row = move["position"][0] + 1
        col = move["position"][1] + 1
        col_word = column_mapping[str(col)]

        if move["direction"] == "horizontal":
            start_point = col_word + str(row)
        else:
            start_point = str(row) + col_word

        # tiles used
        used_tiles_str = "".join(move["tiles_used"]).replace(".", "")
        
        # score increment
        plus_score = move["score"]

        # total score
        if player == "quackle":
            self.quackle_total_score += plus_score
            total_score = self.quackle_total_score
        else:
            self.us_total_score += plus_score
            total_score = self.us_total_score

        insert_sentence = f">{player}: {rack_before_move_str} {start_point} {used_tiles_str} +{plus_score} {total_score}\n"

        # 插入至倒數第二行，加入move record
        with open(self.gcg_name, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 2:
            lines.append(insert_sentence)
        else:
            lines = lines[:-1] + [insert_sentence] + [lines[-1]]

        with open(self.gcg_name, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"add in gcg: {insert_sentence}")


    # 更新quackle 手上現在有的牌
    def update_rack2_line(self, new_rack):
        with open(self.gcg_name, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 找到最後一個以 #rack2 開頭的行
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("#rack2"):
                lines[i] = f"#rack2 {new_rack}\n"
                break
        else:
            # 如果沒找到就加一行
            lines.append(f"#rack2 {new_rack}\n")

        # 寫回檔案
        with open(self.gcg_name, "w", encoding="utf-8") as f:
            f.writelines(lines)

    # think
    def choose_move(
        self, 
        state: Dict, #用不到
        valid_moves: List[Dict], 
        training,
        level = "Speedy Player") :

       # update latest rack
        first_move = valid_moves[0]
       
        remaining_rack = first_move["remaining_rack"]
        tiles_used = first_move["tiles_used"]
        rack_before_move = tiles_used + remaining_rack
        rack_before_move_str = "".join(rack_before_move).replace(".", "")

        self.update_rack2_line( rack_before_move_str)
       
       
        # "[first position] [word]"
        move = quackle_py.get_quackle_answer(
            data_path=self.gcg_name,
            player=level
        )

        print("move", move)

        # 轉成move 格式
        move = self.parse_quackle_output(move["move"], rack_before_move)

        return move 

    def parse_quackle_output(self, input_str: str, rack_before_move: List) -> Optional[Dict]:
        """
        解析 Quackle 的輸出，例如 "H8 DeWOOl"
        小寫字母代表盤面上原有字母，不從 rack 扣除，會轉為 '.'。
        """
        reverse_col_mapping = {v: int(k) - 1 for k, v in column_mapping.items()}

        try:
            pos_str, raw_word = input_str.strip().split()
        except ValueError:
            return None

        if not pos_str or not raw_word:
            return None

        # 將小寫字母轉為 '.'，大寫保留
        converted_word = ''.join([c if c.isupper() else '.' for c in raw_word])

        # 判斷方向
        if pos_str[0].isalpha():
            # 垂直
            col_letter = pos_str[0].upper()
            row_num = int(pos_str[1:])
            direction = "V"
            col0 = reverse_col_mapping.get(col_letter)
            row0 = row_num - 1
            if col0 is None or not (0 <= row0 < 15):
                return None
            positions = [(row0 + i, col0) for i in range(len(raw_word))]
        elif pos_str[0].isdigit():
            # 水平
            i = 0
            while i < len(pos_str) and pos_str[i].isdigit():
                i += 1
            row_num = int(pos_str[:i])
            col_letter = pos_str[i:].upper()
            direction = "H"
            col0 = reverse_col_mapping.get(col_letter)
            row0 = row_num - 1
            if col0 is None or not (0 <= row0 < 15):
                return None
            positions = [(row0, col0 + i) for i in range(len(raw_word))]
        else:
            return None

        return self.move_generator._create_move_dict(
            word=converted_word,
            positions=positions,
            direction=direction,
            original_rack=rack_before_move
        )



if __name__ == "__main__":

    quackle_agent = QuackleAgent()

    us_movement = {
            'word': "TEST",
            'positions': [(7,6), (7,7), (7,8), (7,9)],
            'direction': "horizontal",
            'tiles_used': ['T', 'E', 'S', 'T'],
            'remaining_rack': ["A", "B", "C", "D"],
            'score': 16,
            'premium_squares_used': 16,
            'word_length': 4,
            'position': (7,6),
            
            # Additional info for feature extraction
            'adjacent_empty_squares': 8,  # Approximation
            'premium_squares_blocked': 0,  # Simplified
            'word_hooks_created': 0,
            'board_congestion_increase': 4 // 3
        }

    quackle_agent.to_quackle_input("us", us_movement) 


    valid_movement = [
        {
            'word': ".IGER",
            'positions': [(7,6), (8,6), (9,6), (10,6), (11,6)],
            'direction': "vertical",
            'tiles_used': ["I", "G", "E","R",],
            'remaining_rack': ["F", "F", "L"],
            'score': 9,
            'premium_squares_used': 16,
            'word_length': 5,
            'position': (7,6),
            
            # Additional info for feature extraction
            'adjacent_empty_squares': 8,  # Approximation
            'premium_squares_blocked': 0,  # Simplified
            'word_hooks_created': 0,
            'board_congestion_increase': 5 // 3
        }
    ]

    quackle_choose = quackle_agent.choose_move(
        state={},
        valid_moves=valid_movement,
        training=True
    )
    print("this is what you get from choose move", quackle_choose)
    if quackle_choose is None:
        print("No move")
    else: 
        print("this is what you get from choose move", quackle_choose)
        quackle_agent.to_quackle_input("quackle", quackle_choose) 
    


    


        

