"""
game_stats.py - Modul za trajno praćenje i pohranu globalnih rezultata igara po razinama težine
i početnom igraču (tko je prvi na potezu: IGRAČ ili ROBOT).

Ukupno 18 kategorija (3 težine x 2 početna igrača x 3 ishoda):
- EASY:   PLAYER FIRST (Wins, Draws, Loses) | ROBOT FIRST (Wins, Draws, Loses)
- MEDIUM: PLAYER FIRST (Wins, Draws, Loses) | ROBOT FIRST (Wins, Draws, Loses)
- HARD:   PLAYER FIRST (Wins, Draws, Loses) | ROBOT FIRST (Wins, Draws, Loses)

Napomena: 'Wins' i 'Loses' se odnose na AGENTA (ROBOTA).
- Robot pobijedio: Wins +1
- Igrač pobijedio: Loses +1
- Neriješeno: Draws +1
"""

import os
import re
import config

DEFAULT_DIFFICULTIES = ['EASY', 'MEDIUM', 'HARD']
DEFAULT_FIRST_MOVES = ['PLAYER_FIRST', 'ROBOT_FIRST']


def get_default_stats():
    """Vraća zadanu praznu strukturu statistike za 3 težine i 2 početna igrača."""
    return {
        diff: {
            'PLAYER_FIRST': {'Wins': 0, 'Draws': 0, 'Loses': 0},
            'ROBOT_FIRST': {'Wins': 0, 'Draws': 0, 'Loses': 0},
            # Agregirani zbrojevi za kompatibilnost s postojećim web semaforom i pregledima
            'Wins': 0,
            'Draws': 0,
            'Loses': 0
        }
        for diff in DEFAULT_DIFFICULTIES
    }


def load_stats(filepath=None):
    """
    Učitava statistiku iz tekstualne datoteke.
    Podržava i novi format s podjelom na PLAYER FIRST / ROBOT FIRST,
    kao i stari format radi unakrsne kompatibilnosti.
    Ako datoteka ne postoji, kreira je s početnim vrijednostima (samo ako je RECORD_GAME_STATS True).
    """
    if filepath is None:
        filepath = getattr(config, 'GAME_STATS_FILE', os.path.join("game_results", "game_stats.txt"))

    stats = get_default_stats()

    if not os.path.exists(filepath):
        if getattr(config, 'RECORD_GAME_STATS', False):
            save_stats(stats, filepath)
        return stats


    current_diff = None
    current_first = 'PLAYER_FIRST'

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('#'):
                    continue

                upper_line = line_str.upper()

                # Provjera naziva težine (EASY, MEDIUM, HARD)
                if upper_line in DEFAULT_DIFFICULTIES:
                    current_diff = upper_line
                    current_first = 'PLAYER_FIRST'
                    continue

                # Provjera podsekcije tko je prvi na potezu
                if 'PLAYER' in upper_line and ('FIRST' in upper_line or 'PRVI' in upper_line):
                    current_first = 'PLAYER_FIRST'
                    continue
                elif 'ROBOT' in upper_line and ('FIRST' in upper_line or 'PRVI' in upper_line):
                    current_first = 'ROBOT_FIRST'
                    continue

                # Provjera redaka s ključem i vrijednošću (npr. Wins: 3)
                if current_diff and ':' in line_str:
                    key_part, val_part = line_str.split(':', 1)
                    key_clean = key_part.strip().capitalize()
                    # Normalizacija naziva ishoda
                    if key_clean in ('Losses', 'Loses', 'Loss', 'Lose'):
                        key_clean = 'Loses'
                    elif key_clean in ('Wins', 'Win'):
                        key_clean = 'Wins'
                    elif key_clean in ('Draws', 'Draw', 'Ties', 'Tie'):
                        key_clean = 'Draws'

                    match = re.search(r'-?\d+', val_part)
                    if match:
                        try:
                            val_num = max(0, int(match.group()))
                            if key_clean in ('Wins', 'Draws', 'Loses'):
                                stats[current_diff][current_first][key_clean] = val_num
                        except ValueError:
                            pass

        # Ažuriraj ukupne zbrojeve po svakoj težini
        for diff in DEFAULT_DIFFICULTIES:
            for outcome in ('Wins', 'Draws', 'Loses'):
                stats[diff][outcome] = (
                    stats[diff]['PLAYER_FIRST'][outcome] +
                    stats[diff]['ROBOT_FIRST'][outcome]
                )
    except Exception as e:
        print(f"[WARNING] Error reading statistics from '{filepath}': {e}. Using default values.")

    return stats


def save_stats(stats, filepath=None):
    """
    Sprema statistiku u tekstualnu datoteku u preglednom formatu:
    Za svaku težinu zapisuje PLAYER FIRST (Wins, Draws, Loses) i ROBOT FIRST (Wins, Draws, Loses).
    Ako je RECORD_GAME_STATS False, datoteka se ne mijenja.
    """
    if filepath is None:
        filepath = getattr(config, 'GAME_STATS_FILE', os.path.join("game_results", "game_stats.txt"))

    if not getattr(config, 'RECORD_GAME_STATS', False):
        return False
    try:
        dir_path = os.path.dirname(os.path.abspath(filepath))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, diff in enumerate(DEFAULT_DIFFICULTIES):
                diff_data = stats.get(diff, {})
                pf = diff_data.get('PLAYER_FIRST', {'Wins': 0, 'Draws': 0, 'Loses': 0})
                rf = diff_data.get('ROBOT_FIRST', {'Wins': 0, 'Draws': 0, 'Loses': 0})

                f.write(f"{diff}\n")
                f.write("PLAYER FIRST:\n")
                f.write(f"Wins: {pf.get('Wins', 0)}\n")
                f.write(f"Draws: {pf.get('Draws', 0)}\n")
                f.write(f"Loses: {pf.get('Loses', 0)}\n\n")

                f.write("ROBOT FIRST:\n")
                f.write(f"Wins: {rf.get('Wins', 0)}\n")
                f.write(f"Draws: {rf.get('Draws', 0)}\n")
                f.write(f"Loses: {rf.get('Loses', 0)}\n")

                if i < len(DEFAULT_DIFFICULTIES) - 1:
                    f.write("\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save statistics to '{filepath}': {e}")
        return False


def record_game_result(difficulty, winner, first_move='PLAYER', filepath=None):
    """
    Zapisuje ishod završene igre u trajnu datoteku.
    - difficulty: 'EASY', 'MEDIUM', 'HARD'
    - winner: 'ROBOT', 'PLAYER', 'DRAW'
    - first_move: 'PLAYER' ili 'ROBOT'
    Vraća: (updated_stats, log_message: str)
    """
    if filepath is None:
        filepath = getattr(config, 'GAME_STATS_FILE', os.path.join("game_results", "game_stats.txt"))

    diff_key = str(difficulty).upper()
    if diff_key not in DEFAULT_DIFFICULTIES:
        diff_key = 'HARD'

    first_key = 'ROBOT_FIRST' if 'ROBOT' in str(first_move).upper() else 'PLAYER_FIRST'
    first_label = "ROBOT FIRST" if first_key == 'ROBOT_FIRST' else "PLAYER FIRST"

    # Učitaj najnovije stanje (u slučaju da je korisnik ručno editirao datoteku)
    stats = load_stats(filepath)

    if not getattr(config, 'RECORD_GAME_STATS', False):
        return stats, f"[STATISTICS] Result recording is disabled (RECORD_GAME_STATS=False). Outcome: {winner}, {first_label}."


    outcome_str = ""
    if winner == 'ROBOT':
        stats[diff_key][first_key]['Wins'] += 1
        outcome_str = "ROBOT WIN (Wins +1)"
    elif winner == 'PLAYER':
        stats[diff_key][first_key]['Loses'] += 1
        outcome_str = "ROBOT LOSS (Loses +1)"
    else:  # 'DRAW'
        stats[diff_key][first_key]['Draws'] += 1
        outcome_str = "DRAW (Draws +1)"

    # Ažuriraj ukupne zbrojeve po težini
    for outcome in ('Wins', 'Draws', 'Loses'):
        stats[diff_key][outcome] = (
            stats[diff_key]['PLAYER_FIRST'][outcome] +
            stats[diff_key]['ROBOT_FIRST'][outcome]
        )

    save_stats(stats, filepath)

    current_data = stats[diff_key][first_key]
    msg = (
        f"[STATISTICS - {diff_key} ({first_label})] {outcome_str} | "
        f"Category: Wins: {current_data['Wins']}, Draws: {current_data['Draws']}, Loses: {current_data['Loses']} | "
        f"Total ({diff_key}): Wins: {stats[diff_key]['Wins']}, Draws: {stats[diff_key]['Draws']}, Loses: {stats[diff_key]['Loses']}"
    )
    return stats, msg

