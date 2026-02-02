import serial
import time

#ssc32 = serial.Serial('/dev/ttyS0', 115200)
ssc32 = serial.Serial('COM1', 115200)

home_position = {
    "base": 1550,
    "shoulder": 1450,
    "elbow": 1450,
    "wrist": 1500,
    "rotate": 1400,
    "gripEmpty": 1000,
    "gripHolding": 1300
}

current_position = {
    "base": 1550,
    "shoulder": 1450,
    "elbow": 1450,
    "wrist": 1500,
    "rotate": 1400,
    "grip": 1000,
}

fields = {
    "base": "#0 P",
    "shoulder": "#1 P",
    "elbow": "#2 P",
    "wrist": "#3 P",
    "rotate": "#4 P",
    "grip": "#5 P"
}

def update_position(field, newVal):
    current_position[field] = newVal

def make_command(new_position):
    sequence = [
            (fields['base'], new_position['base']),
            (fields['elbow'], new_position['elbow']),
            (fields['shoulder'], new_position['shoulder']),
            (fields['wrist'], new_position['wrist']),
            (fields['rotate'], new_position['rotate']),
            (fields['grip'], new_position['grip']),
    ]
    
    for field, value in sequence:
        if new_position[field] != current_position[field]:
            send(f"{field}{value} T{int(1000)}")
        time.sleep(1)
    current_position = new_position


def send(command):
    ssc32.write( (command+"\r").encode() )

def home(grip_loaded=False, duration_ms=1000):
    grip_value = home_position["gripHolding"] if grip_loaded else home_position["gripEmpty"]
    sequence = [
        (fields['rotate'], home_position['rotate']),
        (fields['wrist'], home_position['wrist']),
        (fields['shoulder'], home_position['shoulder']),
        (fields['elbow'], home_position['elbow']),
        (fields['base'], home_position['base']),
        (fields['grip'], grip_value),
    ]

    for field, value in sequence:
        send(f"{field}{value} T{int(duration_ms)}")
        time.sleep(1)

    if(blocksLeft == 4):
        sequence = [
            # Move to above
            (fields['base'], ),
            (fields['elbow'], ),
            (fields['shoulder'], ),
            (fields['wrist'], ),
            (fields['rotate'], ),
            # Open Grip
            (fields['grip'], 1000),
            # Lower to block

            # TODO Lower down arm
            
            # Grip the block
            (fields['grip'], 1400),
        ]
    elif(blocksLeft == 3):
        # TODO Add in other block positions
        sequence = [
            (fields['base'], ),
        ]
        
    


def main():
    blocksLeft = 4
    board = [['-' for _ in range(3)] for _ in range(3)]

    print("Does the AI play X? 1 - yes 0 - no")
    ai_symbol = 'x' if input() == '1' else 'o'
    human_symbol = 'o' if ai_symbol == 'x' else 'x'
    
    turn  = True if ai_symbol == 'x' else False
    
    home()
    
    while True:
        winner = detect_win(board)
        
        if winner or tie(board):
            break

        if turn:
            x, y = best_move(board, ai_symbol)
            set_position(x, y, ai_symbol, board)
            print_board(board)
            pickup(blocksLeft)
        else:
            while True:
                raw = input("Enter your move as row col (0 0): ").strip()
                try:
                    r_str, c_str = raw.split()
                    r, c = int(r_str), int(c_str)
                    if r not in range(3) or c not in range(3):
                        print("Coords must be 0,1,2. Try again.")
                        continue
                    if board[r][c] != '-':
                        print("That spot is taken. Try again.")
                        continue
                    set_position(r, c, human_symbol, board)
                    print_board(board)
                    break
                except ValueError:
                    print("Format is: row col (two numbers). Try again.")
                    continue
        turn = not turn

    winner = detect_win(board)
    if winner:
        print(f"{winner.upper()} wins!")
    else:
        print("It's a tie.")


def print_board(board):
    for row in board:
        print(' '.join(row))
    print()


def set_position(x, y, player, board):
    board[x][y] = player


def best_move(board, ai_symbol):
    """Return coordinates (x, y) of the best move for ai_symbol."""
    opponent = 'o' if ai_symbol == 'x' else 'x'
    best_score = float('-inf')
    move = (-1, -1)

    move_order = [
        (1, 1),  # center
        (0, 0), (0, 2), (2, 0), (2, 2),  # corners
        (0, 1), (1, 0), (1, 2), (2, 1)   # edges
    ]

    for i, j in move_order:
        if board[i][j] == '-':
            board[i][j] = ai_symbol
            score = minimax(board, 0, False, ai_symbol, opponent)
            board[i][j] = '-'
            if score > best_score:
                best_score = score
                move = (i, j)
    return move


def evaluate(board):
    """Return board score from X's perspective: 10 win, -10 loss, 0 otherwise."""
    winner = detect_win(board)
    if winner == 'x':
        return 10
    if winner == 'o':
        return -10
    return 0

def tie(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                return False
    return True

def detect_win(board):
    # Rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '-':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != '-':
            return board[0][i]
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] != '-':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '-':
        return board[0][2]
    return None


def minimax(board, depth, is_maximizing, ai_symbol, opponent):
    score = evaluate(board)
    # Flip perspective when AI plays 'o'
    if ai_symbol == 'o':
        score = -score

    if score == 10:
        return score - depth
    if score == -10:
        return score + depth
    if tie(board):
        return 0

    if is_maximizing:
        best = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    board[i][j] = ai_symbol
                    best = max(best, minimax(board, depth + 1, False, ai_symbol, opponent))
                    board[i][j] = '-'
        return best

    best = float('inf')
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                board[i][j] = opponent
                best = min(best, minimax(board, depth + 1, True, ai_symbol, opponent))
                board[i][j] = '-'
    return best

main()
