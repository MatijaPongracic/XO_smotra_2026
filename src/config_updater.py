"""
config_updater.py - Modul za sigurno ažuriranje parametara u radnoj memoriji i u datoteci config.py.
Omogućuje trenutnu primjenu novih parametara tijekom rada aplikacije (bez restarta)
i trajno spremanje u datoteku config.py na disku.
"""

import os
import re
import sys
import config
from src.robot import connect_robot


def update_config_values(updates: dict, config_path: str = None) -> tuple[bool, str]:
    """
    Ažurira zadane parametre:
    1. U radnoj memoriji (moduli config i connect_robot).
    2. U izvornoj datoteci config.py na disku, čuvajući postojeću strukturu i komentare.

    updates: rječnik oblika {'J1': -190.0, 'DRAW_DEPTH': 6.0, ...}
    Vraća: (success: bool, message: str)
    """
    if config_path is None or config_path == "config.py":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.py")

    if not os.path.exists(config_path):
        return False, f"File '{config_path}' not found!"

    # 1. Ažuriranje u radnoj memoriji
    for key, val in updates.items():
        try:
            setattr(config, key, val)
            if hasattr(connect_robot, key):
                setattr(connect_robot, key, val)
        except Exception as e:
            return False, f"Error updating memory for '{key}': {e}"

    # 2. Ažuriranje u datoteci config.py
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        updated_keys = set()

        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            replaced = False
            for key, val in updates.items():
                pattern = rf"^(\s*{re.escape(key)}\s*=\s*)([^\s#]+)(\s*)(#.*)?$"
                match = re.match(pattern, line)
                if match:
                    prefix = match.group(1)
                    comment = match.group(4) if match.group(4) else ""
                    space_before_comment = match.group(3) if (match.group(3) and comment) else ("   " if comment else "")
                    if isinstance(val, float):
                        val_str = f"{val:.2f}" if abs(val - round(val, 1)) > 1e-4 else f"{val:.1f}"
                    else:
                        val_str = str(val)
                    new_line = f"{prefix}{val_str}{space_before_comment}{comment}\n"
                    new_lines.append(new_line)
                    updated_keys.add(key)
                    replaced = True
                    break
            if not replaced:
                new_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")


        with open(config_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(new_lines)

        not_found = set(updates.keys()) - updated_keys
        if not_found:
            msg = f"Updated {len(updated_keys)} parameters in memory and config.py. (Not found in file: {list(not_found)})"
        else:
            msg = f"Successfully updated parameters ({', '.join(updates.keys())}) in memory and config.py file."
        return True, msg

    except Exception as e:
        return False, f"Error writing to '{config_path}': {e}"
