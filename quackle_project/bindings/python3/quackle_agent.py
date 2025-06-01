# coding: utf-8

import os
import ctypes

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

#把執行環境位置固定在本檔案位置，下面的西對路徑才不會出錯
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 取得目前這個 Python 檔案所在的資料夾
current_dir = os.path.dirname(os.path.abspath(__file__))

# 組出兩個 .so 的完整絕對路徑
libquackle_path = os.path.abspath(os.path.join(current_dir, "../../quacker/build/libquackle/liblibquackle.so"))
quackleio_path = os.path.abspath(os.path.join(current_dir, "../../quacker/build/quackleio/libquackleio.so"))

# 載入這些共享庫，順序很重要：liblibquackle 要先
ctypes.CDLL(libquackle_path, mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(quackleio_path, mode=ctypes.RTLD_GLOBAL)

import quackle


def startUp(lexicon='twl06',
            alphabet='english',
            datadir='../../data',
            is_bonus_table = True ):

    # Set up the data manager
    dm = quackle.DataManager()
    dm.setComputerPlayers(quackle.ComputerPlayerCollection.fullCollection())
    dm.setBackupLexicon(lexicon)
    dm.setAppDataDirectory(datadir)

    # Set up the alphabet
    abc = quackle.AlphabetParameters.findAlphabetFile(alphabet)
    abc2 = quackle.Util.stdStringToQString(abc) #convert to qstring
    fa = quackle.FlexibleAlphabetParameters()

    assert fa.load(abc2)
    dm.setAlphabetParameters(fa)

    if not is_bonus_table:
        # Set up the board with extra value 
        board = quackle.BoardParameters()
        dm.setBoardParameters(board)

    else: 
    # Set up the board without extra value
        board = quackle.BoardParameters()
        for row in range(15):
            for col in range(15):
                board.setLetterMultiplier(row, col, 1)  # 字母倍率 1x
                board.setWordMultiplier(row, col, 1)    # 單詞倍率 1x
        dm.setBoardParameters(board)


    # Find the lexicon
    dawg = quackle.LexiconParameters.findDictionaryFile(lexicon + '.dawg')
    gaddag = quackle.LexiconParameters.findDictionaryFile(lexicon + '.gaddag')
    dm.lexiconParameters().loadDawg(dawg)
    dm.lexiconParameters().loadGaddag(gaddag)

    dm.strategyParameters().initialize(lexicon)
    return dm


def getComputerPlayer(dm, name='Championship Player'):
    player, found = dm.computerPlayers().playerForName(name)
    assert found
    player = player.computerPlayer()
    return player


def get_quackle_answer(
     data_path = '../../test/positions/short_game_with_bad_moves.gcg',
     player = 'Speedy Player',
     is_bonus_table = True
    ) :
    """
        input : 
            data_path: the path of the game file
            player: the name of the computer player
            is_bonus_table: True or False

        output:
            first position(containing direction info) and answer letters
    """

    dm = startUp(is_bonus_table=is_bonus_table)

    # Create a computer player
    player1 = getComputerPlayer(dm, name=player)
    print(player1.name())

    # Create the Game file (.gcg) reader
    gamereader = quackle.GCGIO()
    gamePath = quackle.Util.stdStringToQString(data_path)
    game = gamereader.read(gamePath, quackle.Logania.MaintainBoardPreparation)

    # Get the current position
    position = game.currentPosition()

    player1.setPosition(position)

    racks = quackle.ProbableRackList()
    unseenbag = position.unseenBag()
    if unseenbag.size() <= dm.parameters().rackSize() + 3:
        enum = quackle.Enumerator(unseenbag)
        enum.enumerate(racks)
        for rack in racks:
            print(rack)

    movesToShow = 10

    print("Board state: \n%s" % position.board().toString())
    print("Move made: %s" % position.moveMade().toString())
    print("Current player: %s" % position.currentPlayer().storeInformationToString())
    print("Turn number: %i" % position.turnNumber())

    # player rack beefore move
    before_rack = position.currentPlayer().rack().toString()
    print("before rack", before_rack)

    # player move
    move_set = player1.moves(100)  
    
    selected_move = None

    #跳過所有有.的選項
    for move in move_set:
        move_str = move.toString()
        if '.' not in move_str:
            selected_move = move_str
            break

    return {
        "before_rack": before_rack,
        "move": selected_move,
    }

if __name__ == '__main__':
   test =  get_quackle_answer()
   print(test, "haha")