/**
 * controller.js - Upravljanje mobilnim kontrolerom igre
 */

document.addEventListener('DOMContentLoaded', () => {
  // Stanje odabira postavki
  const settings = {
    player_char: 'X',
    difficulty: 'MEDIUM',
    first_move: 'PLAYER',
    grid_drawer: 'PLAYER',
    robot_grid_size: 'MEDIUM',
    robot_grid_pos: 'CENTER'
  };

  // Sekcije
  const setupSection = document.getElementById('setup-section');
  const activeSection = document.getElementById('active-section');
  const robotGridOptions = document.getElementById('robot-grid-options');
  const robotGridPosHint = document.getElementById('robot-grid-pos-hint');

  // Gumbi
  const btnStart = document.getElementById('btn-start-game');
  const btnReset = document.getElementById('btn-reset-mobile');

  // Elementi aktivne igre
  const liveTurnBadge = document.getElementById('live-turn-badge');
  const liveInstruction = document.getElementById('live-instruction');
  const miniBoard = document.getElementById('mini-board');

  function updateGridPosRestrictions() {
    const isLarge = (settings.robot_grid_size === 'LARGE');
    const posGroup = document.querySelector('.btn-group[data-setting="robot_grid_pos"]');
    if (!posGroup) return;

    const btnUp = posGroup.querySelector('[data-value="UP"]');
    const btnCenter = posGroup.querySelector('[data-value="CENTER"]');
    const btnDown = posGroup.querySelector('[data-value="DOWN"]');

    if (isLarge) {
      settings.robot_grid_pos = 'CENTER';
      if (btnUp) {
        btnUp.disabled = true;
        btnUp.classList.remove('active');
        btnUp.classList.add('disabled');
      }
      if (btnDown) {
        btnDown.disabled = true;
        btnDown.classList.remove('active');
        btnDown.classList.add('disabled');
      }
      if (btnCenter) {
        btnCenter.disabled = false;
        btnCenter.classList.add('active');
        btnCenter.classList.remove('disabled');
      }
      if (robotGridPosHint) {
        robotGridPosHint.style.display = 'block';
      }
    } else {
      if (btnUp) {
        btnUp.disabled = false;
        btnUp.classList.remove('disabled');
      }
      if (btnDown) {
        btnDown.disabled = false;
        btnDown.classList.remove('disabled');
      }
      if (robotGridPosHint) {
        robotGridPosHint.style.display = 'none';
      }
    }
  }

  // Vezivanje odabira za sve .btn-group gumbe
  document.querySelectorAll('.btn-group').forEach(group => {
    const key = group.dataset.setting;
    if (!key) return;

    group.querySelectorAll('.btn-option').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.disabled || btn.classList.contains('disabled')) return;

        group.querySelectorAll('.btn-option').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const val = btn.dataset.value;
        settings[key] = val;

        // Ako se bira tko crta mrežu, prikaži/sakrij dodatne opcije za robota
        if (key === 'grid_drawer') {
          if (val === 'ROBOT') {
            robotGridOptions.style.display = 'flex';
          } else {
            robotGridOptions.style.display = 'none';
          }
        }

        // Ako se odabere veličina mreže, zaključaj poziciju na CENTER u slučaju LARGE mreže
        if (key === 'robot_grid_size') {
          updateGridPosRestrictions();
        }
      });
    });
  });

  // Inicijalna provjera ograničenja pozicije
  updateGridPosRestrictions();

  // Pokretanje igre klikom na START
  btnStart.addEventListener('click', () => {
    btnStart.disabled = true;
    btnStart.textContent = 'STARTING...';

    fetch('/api/start-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    })
      .then(r => r.json())
      .then(data => {
        btnStart.disabled = false;
        btnStart.textContent = '▶ START GAME';
      })
      .catch(err => {
        alert('Error starting game: ' + err);
        btnStart.disabled = false;
        btnStart.textContent = '▶ START GAME';
      });
  });

  // Reset igre s mobitela
  btnReset.addEventListener('click', () => {
    if (confirm('Do you want to abort the game and return to setup options?')) {
      fetch('/api/new-game', { method: 'POST' }).catch(() => {});
    }
  });

  // WebSocket veza za praćenje stanja u stvarnom vremenu
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        updateMobileUI(state);
      } catch (e) {}
    };

    ws.onclose = () => {
      setTimeout(connectWebSocket, 1500);
    };
  }

  connectWebSocket();

  // Brzi inicijalni dohvat stanja preko HTTP-a bez čekanja WebSocket rukovanja
  fetch('/api/state')
    .then(r => r.json())
    .then(data => updateMobileUI(data))
    .catch(() => {});

  // Rezervno periodičko osvježavanje (ako mobitel uspava WebSocket u pozadini)
  setInterval(() => {
    fetch('/api/state')
      .then(r => r.json())
      .then(data => updateMobileUI(data))
      .catch(() => {});
  }, 1500);

  // Trenutno osvježavanje kada se mobitel otključa ili vrati u preglednik
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      fetch('/api/state')
        .then(r => r.json())
        .then(data => updateMobileUI(data))
        .catch(() => {});
    }
  });

  function updateMobileUI(state) {
    if (!state) return;

    if (state.grid_sizes) {
      updateGridSizeButtons(state.grid_sizes);
    }

    if (state.game_active) {
      setupSection.style.display = 'none';
      activeSection.style.display = 'flex';

      const activeTitle = document.getElementById('active-game-title');
      const activeSubtitle = document.getElementById('active-game-subtitle');
      const gameoverBanner = document.getElementById('gameover-mobile-banner');
      const btnReset = document.getElementById('btn-reset-mobile');

      if (state.game_over) {
        if (activeTitle) activeTitle.textContent = '🏁 GAME OVER';
        if (activeSubtitle) activeSubtitle.textContent = 'Game is finished!';
        if (liveTurnBadge) liveTurnBadge.style.display = 'none';

        if (gameoverBanner) {
          gameoverBanner.style.display = 'block';
          if (state.game_winner === 'PLAYER') {
            gameoverBanner.style.background = 'rgba(16, 185, 129, 0.25)';
            gameoverBanner.style.border = '2px solid #10b981';
            gameoverBanner.style.color = '#10b981';
            gameoverBanner.innerHTML = '🏆 PLAYER WINS!<br><span style="font-size: 13px; font-weight: 600; color: #a7f3d0;">Congratulations! You beat the robot!</span>';
          } else if (state.game_winner === 'ROBOT') {
            gameoverBanner.style.background = 'rgba(251, 113, 133, 0.25)';
            gameoverBanner.style.border = '2px solid #fb7185';
            gameoverBanner.style.color = '#fb7185';
            gameoverBanner.innerHTML = '🤖 ROBOT WINS!<br><span style="font-size: 13px; font-weight: 600; color: #fecdd3;">Robot got 3 in a row!</span>';
          } else {
            gameoverBanner.style.background = 'rgba(245, 158, 11, 0.25)';
            gameoverBanner.style.border = '2px solid #f59e0b';
            gameoverBanner.style.color = '#f59e0b';
            gameoverBanner.innerHTML = '🤝 DRAW!<br><span style="font-size: 13px; font-weight: 600; color: #fde68a;">All cells are filled with no winner.</span>';
          }
        }
        if (btnReset) {
          btnReset.textContent = '▶ START NEW GAME';
          btnReset.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
          btnReset.style.color = '#fff';
          btnReset.style.fontWeight = '800';
          btnReset.style.boxShadow = '0 6px 20px rgba(16, 185, 129, 0.4)';
        }
      } else {
        if (activeTitle) activeTitle.textContent = 'GAME IN PROGRESS';
        if (activeSubtitle) activeSubtitle.textContent = 'Watch the board on the table and the live scoreboard!';
        if (liveTurnBadge) liveTurnBadge.style.display = 'block';
        if (gameoverBanner) gameoverBanner.style.display = 'none';
        if (btnReset) {
          btnReset.textContent = '🔄 Abort game / New game';
          btnReset.style.background = '';
          btnReset.style.color = '';
          btnReset.style.fontWeight = '';
          btnReset.style.boxShadow = '';
        }

        if (state.grid_warning) {
          liveInstruction.innerHTML = `${state.grid_warning}<br><span style="font-size:12px;font-weight:600;color:#fca5a5;">Erase the board with a sponge and redraw the grid!</span>`;
          liveInstruction.style.background = 'rgba(239, 68, 68, 0.35)';
          liveInstruction.style.color = '#f87171';
          liveInstruction.style.border = '2px solid #ef4444';
        } else if (state.multiple_moves_detected) {
          liveInstruction.innerHTML = '⚠️ DRAW ONLY 1 MARK!<br><span style="font-size:12px;font-weight:600;color:#fca5a5;">Multiple new marks detected on the board. Erase extra marks to continue!</span>';
          liveInstruction.style.background = 'rgba(239, 68, 68, 0.35)';
          liveInstruction.style.color = '#f87171';
          liveInstruction.style.border = '2px solid #ef4444';
        } else if (state.invalid_cells && state.invalid_cells.length > 0) {
          liveInstruction.innerHTML = '⚠️ INVALID MARK!<br><span style="font-size:12px;font-weight:600;color:#fca5a5;">Erase the wrong mark from the board to continue.</span>';
          liveInstruction.style.background = 'rgba(239, 68, 68, 0.3)';
          liveInstruction.style.color = '#f87171';
          liveInstruction.style.border = '1.5px solid #ef4444';
        } else {
          liveInstruction.textContent = state.instruction || 'Game in progress';
          liveInstruction.style.background = 'rgba(0,0,0,0.3)';
          liveInstruction.style.color = 'var(--color-warn)';
          liveInstruction.style.border = 'none';
        }

        if (state.current_turn === 'PLAYER') {
          liveTurnBadge.textContent = `🎯 CURRENT TURN: PLAYER (${state.player_char})`;
          liveTurnBadge.style.color = 'var(--color-x)';
        } else {
          liveTurnBadge.textContent = `🤖 CURRENT TURN: ROBOT (${state.robot_char})`;
          liveTurnBadge.style.color = 'var(--color-o)';
        }
      }

      // Mini ploča
      renderMiniBoard(state);
    } else {
      setupSection.style.display = 'flex';
      activeSection.style.display = 'none';
      updateGridPosRestrictions();
    }
  }

  function renderMiniBoard(state) {
    miniBoard.innerHTML = '';
    const board = state.board || [['', '', ''], ['', '', ''], ['', '', '']];
    const winningCells = new Set();
    if (state.winning_line) {
      state.winning_line.forEach(([r, c]) => winningCells.add(`${r}-${c}`));
    }
    const invalidCells = new Set();
    if (state.invalid_cells) {
      state.invalid_cells.forEach(item => invalidCells.add(`${item.r}-${item.c}`));
    }
    if (state.multiple_move_cells) {
      state.multiple_move_cells.forEach(item => invalidCells.add(`${item.r}-${item.c}`));
    }

    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const cell = document.createElement('div');
        cell.className = 'board-cell';
        cell.style.aspectRatio = '1 / 1';
        cell.style.width = '100%';
        cell.style.height = '100%';
        cell.style.minHeight = '0';
        cell.style.display = 'flex';
        cell.style.alignItems = 'center';
        cell.style.justifyContent = 'center';
        cell.style.borderRadius = '14px';
        cell.style.position = 'relative';
        cell.style.boxSizing = 'border-box';

        const cellKey = `${r}-${c}`;
        const isWinner = winningCells.has(cellKey);
        const isInvalid = invalidCells.has(cellKey);

        if (isWinner) {
          cell.classList.add('cell-winner');
          cell.style.background = 'rgba(16, 185, 129, 0.35)';
          cell.style.border = '2.5px solid #10b981';
          cell.style.boxShadow = '0 0 25px rgba(16, 185, 129, 0.8), inset 0 0 10px rgba(16, 185, 129, 0.4)';
        } else if (isInvalid) {
          cell.classList.add('cell-invalid');
          cell.style.background = 'rgba(239, 68, 68, 0.35)';
          cell.style.border = '2.5px solid #ef4444';
          cell.style.boxShadow = '0 0 25px rgba(239, 68, 68, 0.8), inset 0 0 10px rgba(239, 68, 68, 0.4)';

          const badge = document.createElement('span');
          badge.className = 'invalid-badge';
          badge.textContent = '!';
          badge.style.display = 'block';
          cell.appendChild(badge);
        } else {
          cell.style.background = '#1e293b';
          cell.style.border = '2px solid rgba(255, 255, 255, 0.25)';
          cell.style.boxShadow = 'inset 0 2px 6px rgba(255, 255, 255, 0.05), 0 4px 12px rgba(0, 0, 0, 0.3)';
        }

        const val = board[r][c];
        if (val === 'X' || val === 'O') {
          const mark = document.createElement('span');
          mark.className = `cell-mark mark-${val.toLowerCase()} visible`;
          mark.textContent = val;
          mark.style.fontSize = '64px';
          mark.style.fontWeight = '900';
          mark.style.opacity = '1';
          mark.style.transform = 'scale(1)';
          if (isWinner) {
            mark.style.textShadow = '0 0 25px #10b981';
          }
          cell.appendChild(mark);
        }
        miniBoard.appendChild(cell);
      }
    }
  }

  function updateGridSizeButtons(gridSizes) {
    if (!gridSizes) return;
    const btnSmall = document.querySelector('[data-setting="robot_grid_size"] [data-value="SMALL"]');
    const btnMedium = document.querySelector('[data-setting="robot_grid_size"] [data-value="MEDIUM"]');
    const btnLarge = document.querySelector('[data-setting="robot_grid_size"] [data-value="LARGE"]');
    if (btnSmall && gridSizes.SMALL_MM) {
      const s = (gridSizes.SMALL_MM / 10);
      btnSmall.textContent = `Small (${Number.isInteger(s) ? s : s.toFixed(1)}x${Number.isInteger(s) ? s : s.toFixed(1)} cm)`;
    }
    if (btnMedium && gridSizes.MEDIUM_MM) {
      const m = (gridSizes.MEDIUM_MM / 10);
      btnMedium.textContent = `Medium (${Number.isInteger(m) ? m : m.toFixed(1)}x${Number.isInteger(m) ? m : m.toFixed(1)} cm)`;
    }
    if (btnLarge && gridSizes.LARGE_MM) {
      const l = (gridSizes.LARGE_MM / 10);
      btnLarge.textContent = `Large (${Number.isInteger(l) ? l : l.toFixed(1)}x${Number.isInteger(l) ? l : l.toFixed(1)} cm)`;
    }
  }
});
