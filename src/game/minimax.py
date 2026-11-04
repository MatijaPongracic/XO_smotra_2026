import copy
import random

# ==========================================
# 1. POMOĆNE FUNKCIJE
# ==========================================

def get_empty_cells(board):
    """Vraća listu koordinata (r, c) svih slobodnih polja na ploči."""
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == ""]


def find_winning_moves(board, player):
    """
    Pronalazi sva slobodna polja na kojima zadani igrač može odmah pobijediti u sljedećem potezu.
    """
    winning_moves = []
    for r, c in get_empty_cells(board):
        board[r][c] = player
        if check_winner(board) == player:
            winning_moves.append((r, c))
        board[r][c] = ""
    return winning_moves


def is_valid_state(board, agent="X", opponent="O"):
    agent_count = sum(row.count(agent) for row in board)
    opponent_count = sum(row.count(opponent) for row in board)
    if abs(agent_count - opponent_count) <= 1:
        return True
    return False


def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != "":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != "":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "":
        return board[0][2]
    return None


def is_draw(board):
    for row in board:
        if "" in row:
            return False
    return True


# ==========================================
# 2. HEURISTIČKO MODELIRANJE PROTIVNIKA
# ==========================================

def calculate_human_appeal(board, r, c, player, opponent):
    """
    Ocjenjuje koliko je određeni potez 'prirodan' ljudskom igraču.
    - Ako je linija blokirana protivnikom (mrtvo polje) = 0 bodova.
    - Ako potez samo otvara praznu liniju = 1 bod.
    - Ako potez stvara 2 u nizu (direktan napad) = 5 bodova.
    """
    appeal = 0

    lines = [
        [board[r][0], board[r][1], board[r][2]],  # Red
        [board[0][c], board[1][c], board[2][c]]   # Stupac
    ]
    if r == c:
        lines.append([board[0][0], board[1][1], board[2][2]])
    if r + c == 2:
        lines.append([board[0][2], board[1][1], board[2][0]])

    for line in lines:
        if opponent not in line:  # Linija NIJE mrtva
            player_count = line.count(player)
            if player_count == 2:
                appeal += 5  # Jako prirodan potez (stvara 2 u nizu)
            elif player_count == 1:
                appeal += 1  # Logičan potez (otvara napad)

    return appeal


# ==========================================
# 3. ALGORITMI ODLUČIVANJA PO TEŽINAMA
# ==========================================

def find_move_easy(board):
    """
    EASY razina:
    Agent igra potpuno nasumično na jedno od slobodnih polja.
    Nema nikakve strategije ni prepoznavanja prijetnji.
    """
    empty_cells = get_empty_cells(board)
    if not empty_cells:
        return None
    return random.choice(empty_cells)


def find_move_medium(board, agent="X"):
    """
    MEDIUM razina:
    1) Ako agent može odmah pobijediti (2 svoja znaka i jedno prazno polje u liniji),
       odigraj u to prazno polje i pobijedi odmah.
    2) U suprotnom, ako protivnik ima 2 svoja znaka i jedno prazno polje u liniji,
       odigraj u to prazno polje (blokiraj izravnu pobjedu protivnika).
    3) U suprotnom, odigraj nasumično na jedno od slobodnih polja.
    """
    opponent = "O" if agent == "X" else "X"

    # 1. Može li agent odmah pobijediti?
    winning_moves = find_winning_moves(board, agent)
    if winning_moves:
        return random.choice(winning_moves)

    # 2. Prijeti li protivnik neposrednom pobjedom (blokada)?
    blocking_moves = find_winning_moves(board, opponent)
    if blocking_moves:
        return random.choice(blocking_moves)

    # 3. Nasumičan odabir slobodnog polja
    empty_cells = get_empty_cells(board)
    if not empty_cells:
        return None
    return random.choice(empty_cells)


def minimax(board, depth, alpha, beta, is_maximizing, agent="X", opponent="O"):
    winner = check_winner(board)

    if winner == agent:
        return 10 - depth
    elif winner == opponent:
        return -10 + depth
    elif is_draw(board):
        return 0

    if is_maximizing:
        max_eval = -float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    board[r][c] = agent
                    eval_score = minimax(board, depth + 1, alpha, beta, False, agent, opponent)
                    board[r][c] = ""
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    board[r][c] = opponent
                    eval_score = minimax(board, depth + 1, alpha, beta, True, agent, opponent)
                    board[r][c] = ""
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
        return min_eval


def find_move_hard(board, agent="X"):
    """
    HARD razina:
    Nepobijedivi algoritam temeljen na Minimaxu s alfa-beta podrezivanjem
    i heurističkim ocjenjivanjem prirodnih zamki.
    """
    opponent = "O" if agent == "X" else "X"
    moves_eval = []

    # KORAK 1: Nalazimo optimalne minimax ocjene za sve moguće poteze
    for r in range(3):
        for c in range(3):
            if board[r][c] == "":
                board[r][c] = agent
                score = minimax(board, 0, -float('inf'), float('inf'), False, agent, opponent)
                board[r][c] = ""
                moves_eval.append({'move': (r, c), 'score': score})

    if not moves_eval:
        return None

    max_score = max(moves_eval, key=lambda x: x['score'])['score']
    best_moves = [m['move'] for m in moves_eval if m['score'] == max_score]

    # KORAK 2: Pronalaženje prirodnih zamki s randomizacijom
    if len(best_moves) > 1:
        best_trap_moves = []
        max_trap_score = -1

        for r, c in best_moves:
            trap_score = 0
            board[r][c] = agent

            for or_r in range(3):
                for or_c in range(3):
                    if board[or_r][or_c] == "":
                        board[or_r][or_c] = opponent

                        res_score = minimax(board, 1, -float('inf'), float('inf'), True, agent, opponent)

                        if res_score > 0:
                            total_depth = 10 - res_score
                            agent_moves = total_depth // 2
                            base_weight = max(0, agent_moves - 1)

                            if base_weight > 0:
                                appeal = calculate_human_appeal(board, or_r, or_c, opponent, agent)
                                trap_score += (base_weight * appeal)

                        board[or_r][or_c] = ""
            board[r][c] = ""

            # Logika za prikupljanje najboljih izjednačenih poteza
            if trap_score > max_trap_score:
                max_trap_score = trap_score
                best_trap_moves = [(r, c)]
            elif trap_score == max_trap_score:
                best_trap_moves.append((r, c))

        return random.choice(best_trap_moves)
    else:
        return best_moves[0]


# ==========================================
# 4. GLAVNO SUČELJE (DISPATCHER)
# ==========================================

def find_best_move(board, agent="X", difficulty="HARD"):
    """
    Pronalazi potez za agenta ('X' ili 'O') ovisno o odabranoj težini:
    - 'EASY': Potpuno nasumičan potez.
    - 'MEDIUM': Pobjeđuje odmah ako može, blokira protivnika ako prijeti, inače nasumično.
    - 'HARD': Nepobijedivi Minimax algoritam s alfa-beta podrezivanjem i zamkama.
    Vraća tuple (r, c) ili None ako nema dostupnih poteza.
    """
    diff = str(difficulty).upper() if difficulty else "HARD"

    if diff == "EASY":
        return find_move_easy(board)
    elif diff == "MEDIUM":
        return find_move_medium(board, agent=agent)
    elif diff == "HARD":
        return find_move_hard(board, agent=agent)
    else:
        # Fallback na HARD za nepoznatu težinu
        return find_move_hard(board, agent=agent)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    test_agent = "X"
    empty_board = [
        ["", "", ""],
        ["", "", ""],
        ["", "", ""]
    ]
    print(f"--- Testiranje prazne ploce s agentom: {test_agent} ---")
    print(f"EASY potez:   {find_best_move(empty_board, agent=test_agent, difficulty='EASY')}")
    print(f"MEDIUM potez: {find_best_move(empty_board, agent=test_agent, difficulty='MEDIUM')}")
    print(f"HARD potez:   {find_best_move(empty_board, agent=test_agent, difficulty='HARD')}")

    # Test MEDIUM - pobjeda u jednom potezu
    win_board = [
        ["X", "X", ""],
        ["O", "", "O"],
        ["", "", ""]
    ]
    print(f"\n--- Test MEDIUM: Prioritet pobjede nad blokadom ---")
    print(f"MEDIUM mora odabrati pobjednicko polje (0, 2): {find_best_move(win_board, agent='X', difficulty='MEDIUM')}")

    # Test MEDIUM - blokada protivnika
    block_board = [
        ["O", "O", ""],
        ["X", "", ""],
        ["", "", ""]
    ]
    print(f"\n--- Test MEDIUM: Blokada protivnikove linije ---")
    print(f"MEDIUM mora blokirati protivnika na (0, 2): {find_best_move(block_board, agent='X', difficulty='MEDIUM')}")


