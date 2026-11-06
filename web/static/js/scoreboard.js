/**
 * scoreboard.js - Logika prikaza i sinkronizacije za veliki digitalni semafor
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elementi sučelja
  const attractScreen = document.getElementById('attract-screen');
  const gameScreen = document.getElementById('game-screen');
  const splashOverlay = document.getElementById('splash-overlay');
  const gameoverModal = document.getElementById('gameover-modal');

  const boardGrid = document.getElementById('board-grid');
  const instructionText = document.getElementById('instruction-text');
  const instructionIcon = document.getElementById('instruction-icon');
  const multiMoveAlert = document.getElementById('multi-move-alert');

  const playerCharLabel = document.getElementById('player-char-label');
  const robotCharLabel = document.getElementById('robot-char-label');
  const playerAvatar = document.getElementById('player-avatar');
  const robotAvatar = document.getElementById('robot-avatar');

  const scorePlayerEl = document.getElementById('score-player');
  const scoreRobotEl = document.getElementById('score-robot');
  const difficultyBadge = document.getElementById('difficulty-badge');

  const gameoverTitle = document.getElementById('gameover-title');
  const gameoverSubtitle = document.getElementById('gameover-subtitle');
  const gameoverIcon = document.getElementById('gameover-icon');
  const countdownProgress = document.getElementById('countdown-progress');
  const countdownSeconds = document.getElementById('countdown-seconds');

  const directUrlSpan = document.getElementById('direct-url');
  const btnResetNow = document.getElementById('btn-reset-now');
  const btnInGameReset = document.getElementById('btn-ingame-reset');
  const btnSoundToggle = document.getElementById('btn-sound-toggle');
  const reconnectOverlay = document.getElementById('reconnect-overlay');

  // Osiguraj da početni QR kod nikada nije iz starog cachea preglednika
  const initialQrImage = document.getElementById('qr-image');
  if (initialQrImage) {
    initialQrImage.src = '/api/qrcode.png?t=' + Date.now();
  }

  // Stanje aplikacije
  let ws = null;
  let lastState = null;
  let countdownTimer = null;
  let countdownRemaining = 10;
  let isSplashActive = false;
  let lastWarningSoundTime = 0;
  let wasDisconnected = false;

  // Inicijalizacija zvuka
  function updateSoundButton() {
    const isMuted = window.soundManager.isMuted();
    if (isMuted) {
      btnSoundToggle.classList.add('muted');
      btnSoundToggle.innerHTML = '🔇 SOUND: OFF';
    } else {
      btnSoundToggle.classList.remove('muted');
      btnSoundToggle.innerHTML = '🔊 SOUND: ON';
    }
  }

  btnSoundToggle.addEventListener('click', () => {
    window.soundManager.toggleMute();
    updateSoundButton();
  });
  updateSoundButton();

  // Fullscreen funkcionalnost
  const btnFullscreenToggle = document.getElementById('btn-fullscreen-toggle');
  if (btnFullscreenToggle) {
    btnFullscreenToggle.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => { });
      } else {
        document.exitFullscreen().catch(() => { });
      }
    });

    document.addEventListener('fullscreenchange', () => {
      if (document.fullscreenElement) {
        btnFullscreenToggle.innerHTML = '🗗 EXIT';
        btnFullscreenToggle.title = 'Exit Fullscreen';
      } else {
        btnFullscreenToggle.innerHTML = '⛶ FULLSCREEN';
        btnFullscreenToggle.title = 'Fullscreen';
      }
    });
  }


  let lastControllerUrl = null;

  // Dohvat adrese za povezivanje (Wi-Fi ili Cloudflare javni tunel)
  function updateConnectionDetails() {
    fetch('/api/ip')
      .then(r => r.json())
      .then(data => {
        if (data.controller_url) {
          if (directUrlSpan) {
            directUrlSpan.textContent = data.controller_url;
          }
          // Ako se URL promijenio na javni Cloudflare HTTPS link, osvježi QR kod
          if (lastControllerUrl && lastControllerUrl !== data.controller_url) {
            const qrImage = document.getElementById('qr-image');
            if (qrImage) {
              qrImage.src = '/api/qrcode.png?t=' + Date.now();
            }
          }
          lastControllerUrl = data.controller_url;
        }
      })
      .catch(() => { });
  }


  // Početni dohvat
  updateConnectionDetails();
  // Brze provjere unutar prvih nekoliko sekundi dok se Cloudflare tunel diže
  setTimeout(updateConnectionDetails, 1500);
  setTimeout(updateConnectionDetails, 3000);
  setTimeout(updateConnectionDetails, 5000);
  setTimeout(updateConnectionDetails, 8000);
  // Periodička provjera dok je prikazan attract početni ekran
  setInterval(() => {
    if (attractScreen && attractScreen.style.display !== 'none') {
      updateConnectionDetails();
    }
  }, 10000);


  // Povezivanje na WebSocket
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WEBSOCKET] Uspješno povezan na server.');
      if (wasDisconnected) {
        // Ako je veza bila prekinuta (aplikacija ponovno upaljena), osvježi stranicu
        // kako bi se prikazala nova sesija i novo generirani QR kod
        console.log('[WEBSOCKET] Server ponovno aktivan! Osvježavam sučelje...');
        window.location.reload();
        return;
      }
      if (reconnectOverlay) {
        reconnectOverlay.classList.remove('active');
      }
    };

    ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        handleStateUpdate(state);
      } catch (e) {
        console.error('[WEBSOCKET ERROR]', e);
      }
    };

    ws.onclose = () => {
      console.warn('[WEBSOCKET] Veza prekinuta. Ponovno povezivanje za 1.5s...');
      wasDisconnected = true;
      if (reconnectOverlay) {
        reconnectOverlay.classList.add('active');
      }
      setTimeout(connectWebSocket, 1500);
    };

    ws.onerror = () => {
      if (ws) ws.close();
    };
  }

  connectWebSocket();

  // Obrada pristiglog stanja igre
  function handleStateUpdate(state) {
    if (!state) return;

    // 1. Detekcija prijelaza u novu igru -> Pokreni "THE GAME BEGINS!" animaciju SAMO kada igra stvarno započne
    if (lastState && !lastState.game_active && state.game_active) {
      triggerSplash(state);
    }


    // 2. Detekcija novih poteza na ploči -> Zvuk poteza
    if (lastState && lastState.game_active && state.game_active && !state.game_over) {
      const lastCount = countBoardMoves(lastState.board);
      const curCount = countBoardMoves(state.board);
      if (curCount > lastCount) {
        const isPlayerMove = state.current_turn === 'ROBOT'; // Ako je sad robot na potezu, upravo je igrao čovjek
        window.soundManager.playMove(isPlayerMove);
      }
    }

    // Zvučno upozorenje za neregularnost (više novih znakova u istom potezu ILI odigran krivi znak ILI neispravna mreža ILI zaostali znakovi)
    const hasPreexisting = Boolean(state.preexisting_marks_detected);
    const lastHadPreexisting = Boolean(lastState && lastState.preexisting_marks_detected);
    const hasMulti = Boolean(state.multiple_moves_detected);
    const lastHadMulti = Boolean(lastState && lastState.multiple_moves_detected);
    const hasInvalid = Boolean(state.invalid_cells && state.invalid_cells.length > 0);
    const lastHadInvalid = Boolean(lastState && lastState.invalid_cells && lastState.invalid_cells.length > 0);
    const hasGridWarning = Boolean(state.grid_warning);
    const lastHadGridWarning = Boolean(lastState && lastState.grid_warning);

    const isNewWarning = (hasPreexisting && !lastHadPreexisting) || (hasMulti && !lastHadMulti) || (hasInvalid && !lastHadInvalid) || (hasGridWarning && !lastHadGridWarning);

    if (isNewWarning) {
      const now = Date.now();
      if (now - lastWarningSoundTime > 4000) {
        lastWarningSoundTime = now;
        if (window.soundManager && window.soundManager.playWarning) {
          window.soundManager.playWarning();
        }
      }
    }

    // 3. Detekcija završetka partije -> Zvuk i 10s tajmer
    if (state.game_over && (!lastState || !lastState.game_over)) {
      handleGameOver(state);
    } else if (!state.game_over && countdownTimer) {
      stopCountdown();
    }

    // Ažuriranje prikaza ovisno o aktivnosti igre
    if (!state.game_active) {
      showAttractScreen();
    } else {
      showGameScreen(state);
    }

    lastState = state;
  }

  function countBoardMoves(board) {
    if (!board) return 0;
    let count = 0;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        if (board[r] && board[r][c] !== '') count++;
      }
    }
    return count;
  }

  // Prikaz početnog ekrana
  function showAttractScreen() {
    attractScreen.style.display = 'flex';
    gameScreen.style.display = 'none';
    gameoverModal.classList.remove('active');
    stopCountdown();
  }

  // Prikaz aktivne igre
  function showGameScreen(state) {
    if (isSplashActive) return; // Dok traje intro splash, pričekaj sekundu

    attractScreen.style.display = 'none';
    gameScreen.style.display = 'flex';

    // Postavke igrača i simbola
    playerCharLabel.textContent = `(${state.player_char})`;
    robotCharLabel.textContent = `(${state.robot_char})`;

    playerAvatar.className = `player-avatar avatar-${state.player_char.toLowerCase()}`;
    playerAvatar.textContent = state.player_char;

    robotAvatar.className = `player-avatar avatar-${state.robot_char.toLowerCase()}`;
    robotAvatar.textContent = state.robot_char;

    const robotSystemLabel = document.getElementById('robot-system-label');
    if (robotSystemLabel && state.robot_model) {
      robotSystemLabel.textContent = `${state.robot_model} System`;
    }

    difficultyBadge.textContent = `DIFFICULTY: ${state.difficulty || 'MEDIUM'}`;

    // Ažuriranje rezultata (omjer pobjeda iz game_stats)
    if (state.stats && state.difficulty && state.stats[state.difficulty]) {
      const curStats = state.stats[state.difficulty];
      scorePlayerEl.textContent = curStats.Loses || 0; // Za robota su to Loses -> igraču su to Wins
      scoreRobotEl.textContent = curStats.Wins || 0;
    }

    // Ažuriranje 3x3 ploče
    renderBoard(state);

    // Ažuriranje donje upute (instruction banner)
    updateInstructionBanner(state);
  }

  // Iscrtavanje i ažuriranje ćelija 3x3 ploče bez ponovnog re-renderiranja DOM-a
  function renderBoard(state) {
    const board = state.board || [['', '', ''], ['', '', ''], ['', '', '']];
    const winningCells = new Set();

    if (state.winning_line) {
      state.winning_line.forEach(([r, c]) => winningCells.add(`${r}-${c}`));
    }

    const invalidMap = new Set();
    if (state.invalid_cells) {
      state.invalid_cells.forEach(item => invalidMap.add(`${item.r}-${item.c}`));
    }
    if (state.multiple_move_cells) {
      state.multiple_move_cells.forEach(item => invalidMap.add(`${item.r}-${item.c}`));
    }
    if (state.preexisting_marks_cells) {
      state.preexisting_marks_cells.forEach(item => invalidMap.add(`${item.r}-${item.c}`));
    }

    // Inicijaliziraj 9 ćelija samo jednom
    if (boardGrid.children.length !== 9) {
      boardGrid.innerHTML = '';
      boardGrid.style.background = '#090e1a';
      boardGrid.style.border = '2.5px solid rgba(56, 189, 248, 0.6)';
      boardGrid.style.borderRadius = '24px';
      boardGrid.style.boxShadow = '0 16px 50px rgba(0, 0, 0, 0.5), 0 0 35px rgba(56, 189, 248, 0.25)';
      boardGrid.style.boxSizing = 'border-box';

      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const cell = document.createElement('div');
          cell.className = 'board-cell';
          cell.dataset.r = r;
          cell.dataset.c = c;
          cell.style.background = '#1e293b';
          cell.style.border = '2px solid rgba(255, 255, 255, 0.28)';
          cell.style.borderRadius = '18px';
          cell.style.boxShadow = 'inset 0 2px 10px rgba(255, 255, 255, 0.06), 0 6px 20px rgba(0, 0, 0, 0.4)';
          cell.style.display = 'flex';
          cell.style.alignItems = 'center';
          cell.style.justifyContent = 'center';
          cell.style.overflow = 'hidden';
          cell.style.boxSizing = 'border-box';

          const badge = document.createElement('span');
          badge.className = 'invalid-badge';
          badge.textContent = '!';
          badge.style.display = 'none';

          const mark = document.createElement('span');
          mark.className = 'cell-mark';

          cell.appendChild(badge);
          cell.appendChild(mark);
          boardGrid.appendChild(cell);
        }
      }
    }

    // Ažuriraj samo ćelije čije se stanje promijenilo (sprječava treperenje i trzanje)
    const cells = boardGrid.children;
    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i];
      const r = parseInt(cell.dataset.r, 10);
      const c = parseInt(cell.dataset.c, 10);
      const cellKey = `${r}-${c}`;
      const badge = cell.querySelector('.invalid-badge');
      const mark = cell.querySelector('.cell-mark');

      // 1. Pobjednička linija
      const isWinner = winningCells.has(cellKey);
      if (cell.classList.contains('cell-winner') !== isWinner) {
        cell.classList.toggle('cell-winner', isWinner);
      }

      // 2. Neispravan znak
      const isInvalid = invalidMap.has(cellKey);
      if (cell.classList.contains('cell-invalid') !== isInvalid) {
        cell.classList.toggle('cell-invalid', isInvalid);
      }
      if (badge && (badge.style.display === 'none') === isInvalid) {
        badge.style.display = isInvalid ? 'block' : 'none';
      }

      // 3. Simbol (X, O ili prazno)
      const currentVal = board[r][c] || '';
      const prevVal = mark.dataset.val || '';

      if (currentVal !== prevVal) {
        mark.dataset.val = currentVal;
        mark.textContent = currentVal;
        mark.className = 'cell-mark';
        if (currentVal === 'X' || currentVal === 'O') {
          // Dinamički izračunaj veličinu simbola tako da ispunjava 70-75% visine ćelije
          const cellSize = cell.clientWidth || cell.offsetHeight || 200;
          const markFontSize = Math.max(75, Math.round(cellSize * 0.72));
          mark.style.fontSize = `${markFontSize}px`;
          mark.style.fontWeight = '900';
          mark.style.lineHeight = '1';

          if (currentVal === 'X') {
            mark.classList.add('mark-x', 'visible');
            mark.style.color = '#38bdf8';
            mark.style.textShadow = '0 0 30px rgba(56, 189, 248, 0.75)';
          } else {
            mark.classList.add('mark-o', 'visible');
            mark.style.color = '#fb7185';
            mark.style.textShadow = '0 0 30px rgba(251, 113, 133, 0.75)';
          }
        } else {
          mark.classList.remove('visible');
        }
      }
    }
  }

  // Automatsko prilagođavanje veličine simbola pri promjeni veličine prozora ili prelasku u fullscreen
  window.addEventListener('resize', () => {
    if (!boardGrid) return;
    const cells = boardGrid.querySelectorAll('.board-cell');
    cells.forEach(cell => {
      const mark = cell.querySelector('.cell-mark');
      if (mark && (mark.textContent === 'X' || mark.textContent === 'O')) {
        const cellSize = cell.clientWidth || cell.offsetHeight || 200;
        mark.style.fontSize = `${Math.max(75, Math.round(cellSize * 0.72))}px`;
      }
    });
  });


  // Ažuriranje statusne trake s posebnim naglaskom na zahtjev "DRAW A 3x3 GRID" i upozorenja
  function updateInstructionBanner(state) {
    const rawInstruction = state.instruction || '';
    instructionText.textContent = rawInstruction;
    instructionText.className = '';

    if (state.grid_warning) {
      if (multiMoveAlert) multiMoveAlert.style.display = 'none';
      instructionIcon.textContent = '⚠️';
      instructionText.textContent = state.grid_warning;
      instructionText.classList.add('instruction-warning');
    } else if (state.preexisting_marks_detected) {
      if (multiMoveAlert) {
        multiMoveAlert.style.display = 'flex';
        const titleEl = document.getElementById('multi-move-title');
        const descEl = document.getElementById('multi-move-desc');
        if (titleEl) titleEl.textContent = 'WARNING: PLEASE ERASE ALL MARKS!';
        if (descEl) descEl.textContent = 'Robot plays first. Pre-existing marks detected inside the grid. Please erase all marks from the board to start.';
      }
      instructionIcon.textContent = '⚠️';
      instructionText.textContent = '⚠️ PLEASE ERASE ALL MARKS INSIDE THE GRID!';
      instructionText.classList.add('instruction-warning');
    } else if (state.multiple_moves_detected) {
      if (multiMoveAlert) {
        multiMoveAlert.style.display = 'flex';
        const titleEl = document.getElementById('multi-move-title');
        const descEl = document.getElementById('multi-move-desc');
        if (titleEl) titleEl.textContent = 'WARNING: DRAW ONLY 1 MARK!';
        if (descEl) descEl.textContent = 'Multiple new marks detected on the board. Erase the extra mark with an eraser to continue the game.';
      }
      instructionIcon.textContent = '⚠️';
      instructionText.textContent = state.instruction || '⚠️ WARNING: DRAW ONLY 1 MARK!';
      instructionText.classList.add('instruction-warning');
    } else {
      if (multiMoveAlert) multiMoveAlert.style.display = 'none';

      if (rawInstruction.includes('DRAW A 3x3 GRID') || rawInstruction.includes('NACRTAJTE MREŽU 3x3')) {
        instructionIcon.textContent = '✍️';
        instructionText.classList.add('instruction-draw-grid');
      } else if (rawInstruction.includes('ROBOT IS DRAWING') || rawInstruction.includes('ROBOT CRTA')) {
        instructionIcon.textContent = '🤖';
      } else if (rawInstruction.includes('HAND') || rawInstruction.includes('ŠAKA')) {
        instructionIcon.textContent = '✋';
      } else if (rawInstruction.includes('WINS') || rawInstruction.includes('VICTORY') || rawInstruction.includes('DRAW') || rawInstruction.includes('POBJEDA')) {
        instructionIcon.textContent = '🏆';
      } else if (rawInstruction.includes('PLAYER\'S TURN') || rawInstruction.includes('POTEZ IGRAČA')) {
        instructionIcon.textContent = '🎯';
      } else if (rawInstruction.includes('NEISPRAVAN') || rawInstruction.includes('INVALID') || rawInstruction.includes('UPOZORENJE') || rawInstruction.includes('WARNING')) {
        instructionIcon.textContent = '⚠️';
        instructionText.classList.add('instruction-warning');
      } else {
        instructionIcon.textContent = '⚡';
      }
    }
  }

  // Intro Splash ("THE GAME BEGINS!")
  function triggerSplash(state) {
    isSplashActive = true;
    splashOverlay.classList.add('active');
    window.soundManager.playGameStart();

    setTimeout(() => {
      splashOverlay.classList.remove('active');
      isSplashActive = false;
      if (lastState) {
        showGameScreen(lastState);
      }
    }, 1800);
  }

  // Rukovanje krajem igre s tajmerom od 10 sekundi
  function handleGameOver(state) {
    gameoverModal.classList.add('active');
    const winner = state.game_winner;

    if (winner === 'PLAYER') {
      gameoverIcon.textContent = '🏆';
      gameoverTitle.className = 'gameover-title win-player';
      gameoverTitle.textContent = 'PLAYER WINS!';
      gameoverSubtitle.textContent = `Congratulations! You beat the robot on ${state.difficulty} difficulty!`;
      window.soundManager.playVictory();
    } else if (winner === 'ROBOT') {
      gameoverIcon.textContent = '🤖';
      gameoverTitle.className = 'gameover-title win-robot';
      gameoverTitle.textContent = 'ROBOT WINS!';
      gameoverSubtitle.textContent = 'Better luck next time!';
      window.soundManager.playRobotWin();
    } else {
      gameoverIcon.textContent = '🤝';
      gameoverTitle.className = 'gameover-title win-draw';
      gameoverTitle.textContent = 'DRAW!';
      gameoverSubtitle.textContent = 'All cells are filled.';
      window.soundManager.playDraw();
    }

    startCountdown(10);
  }

  // Odbrojavanje 10 sekundi i automatski reset
  function startCountdown(seconds) {
    stopCountdown();
    countdownRemaining = seconds;
    updateCountdownUI();

    countdownTimer = setInterval(() => {
      countdownRemaining--;
      updateCountdownUI();

      if (countdownRemaining <= 0) {
        stopCountdown();
        triggerNewGame();
      }
    }, 1000);
  }

  function updateCountdownUI() {
    countdownSeconds.textContent = countdownRemaining;
    const pct = (countdownRemaining / 10) * 100;
    countdownProgress.style.width = `${pct}%`;
  }

  function stopCountdown() {
    if (countdownTimer) {
      clearInterval(countdownTimer);
      countdownTimer = null;
    }
  }

  function triggerNewGame() {
    fetch('/api/new-game', { method: 'POST' }).catch(() => { });
  }

  btnResetNow.addEventListener('click', () => {
    stopCountdown();
    triggerNewGame();
  });

  btnInGameReset.addEventListener('click', () => {
    if (confirm('Do you want to abort the current game and return to the main menu?')) {
      triggerNewGame();
    }
  });
});
