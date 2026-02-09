import serial
import time
import locations
#ssc32 = serial.Serial('/dev/ttyS0', 115200)
ssc32 = serial.Serial('COM4', 115200)
do_logging = True

current_position = {
    "base": 1550,
    "shoulder": 1450,
    "elbow": 1450,
    "wrist": 1500,
    "rotate": 1400,
    "grip": 1000,
}

new_position = {
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

def log(msg):
    if do_logging: print(msg)


def open_grip():
    global current_position
    if current_position['grip'] != 1000:
        new_position['grip'] = 1000
        make_command(new_position)
        current_position['grip'] = new_position['grip']
        #current_position = new_position
        log("Opened claw.")
    else:
        log("Cannot open claw, claw already opened.")

def close_grip():
    global current_position
    if current_position['grip'] != 1400:
        new_position['grip'] = 1400
        make_command(new_position)
        current_position['grip'] = new_position['grip']
        #current_position = new_position
        log("Closed claw.")
    else:
        log("Cannot close claw, claw already closed.")

def make_command(new_pos, duration_ms=600):
    order = ["base", "shoulder", "wrist", "rotate", "grip"]
    global current_position

    for joint in order:
        if joint == 'shoulder':
            current_position['elbow'] = new_pos['elbow']
        #desired = new_pos[joint]
        current_position[joint] = new_pos[joint]
        output = f"{fields['base']}{current_position['base']} {fields['shoulder']}{current_position['shoulder']} {fields['elbow']}{current_position['elbow']} {fields['wrist']}{current_position['wrist']} {fields['rotate']}{current_position['rotate']} {fields['grip']}{current_position['grip']} T{int(duration_ms)}"
        send(output)
        #_ = input()
        time.sleep((duration_ms / 1000.0) * 1.2) 

def send(command):
    ssc32.write( (command+"\r").encode() )
    ssc32.flush()

def home(duration_ms=600):
    sequence = [
        ('rotate', locations.home_position['rotate']),
        ('wrist', locations.home_position['wrist']),
        ('shoulder', locations.home_position['shoulder']),
        ('base', locations.home_position['base']),
    ]

    for field, value in sequence:
        if field == 'shoulder':
            new_position['elbow'] = locations.home_position['elbow']
            current_position['elbow'] = locations.home_position['elbow']
        new_position[field] = value
        current_position[field] = value
        output = f"{fields['base']}{current_position['base']} {fields['shoulder']}{current_position['shoulder']} {fields['elbow']}{current_position['elbow']} {fields['wrist']}{current_position['wrist']} {fields['rotate']}{current_position['rotate']} {fields['grip']}{current_position['grip']} T{int(duration_ms)}"
        send(output)            
        #_ = input()
        time.sleep((duration_ms / 1000.0) * 1.2) 

def move_to(location):
    global new_position
    new_position['base'] = location['base']
    new_position['elbow'] = location['elbow']
    new_position['rotate'] = location['rotate']
    new_position['shoulder'] = location['shoulder']
    new_position['wrist'] = location['wrist']
    make_command(new_position)

def pickup_next_block(blocksLeft):
    home()
    if(blocksLeft == 5): move_to(locations.first_cube)
    elif(blocksLeft == 4): move_to(locations.second_cube)
    elif(blocksLeft == 3): move_to(locations.third_cube)
    elif(blocksLeft == 2): move_to(locations.fourth_cube)
    elif(blocksLeft == 1): move_to(locations.fifth_cube)
    else: return # No block to pick up, do nothing.

    close_grip()
    home()
    

def main():
    blocksLeft = 5
    board = [['-' for _ in range(3)] for _ in range(3)]

    print("Does the AI play X? 1 - yes 0 - no")
    ai_symbol = 'x' if input() == '1' else 'o'
    human_symbol = 'o' if ai_symbol == 'x' else 'x'
    
    turn  = True if ai_symbol == 'x' else False
    
    open_grip()
    home()
    
    while True:
        winner = detect_win(board)
        
        if winner or tie(board):
            break

        if turn:
            x, y = best_move(board, ai_symbol)
            set_position(x, y, ai_symbol, board)
            print_board(board)
            
            # Get Next Block
            pickup_next_block(blocksLeft)
            blocksLeft -=1
            
            # Move to Location
            move_to(locations.board_locations[x][y])
	
            #time.sleep(4) # nap time
            # Drop block
            open_grip()
            
            # Return home
            home()
            
            
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
