/**
 * lights.js - Dedicated Visual & Audio Traffic Light for Tablet/Phone.
 * 100% offline Web Audio API sound synthesis and real-time WebSocket state management.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const ambientGlow = document.getElementById('ambient-glow');
  const lightRed = document.getElementById('light-red');
  const lightGreen = document.getElementById('light-green');
  const instructionCard = document.getElementById('instruction-card');
  const instructionMain = document.getElementById('instruction-main');
  const instructionSub = document.getElementById('instruction-sub');
  const connDot = document.getElementById('conn-dot');
  const connText = document.getElementById('conn-text');
  const badgeTurn = document.getElementById('badge-turn');
  const badgeMotion = document.getElementById('badge-motion');
  const btnSoundToggle = document.getElementById('btn-sound-toggle');
  const btnFullscreenToggle = document.getElementById('btn-fullscreen-toggle');
  const startOverlay = document.getElementById('start-overlay');

  // Internal state tracking
  let isMuted = localStorage.getItem('xo_lights_muted') === 'true';
  let wakeLock = null;
  let ws = null;
  let lastSignalState = null; // 'OFF', 'RED', 'GREEN', 'ALARM'
  let lastAlarmSoundTime = 0;
  let motionHandFrames = 0;

  // Web Audio Context
  let audioCtx = null;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  // Synthesize audio tones using Web Audio API
  function playTone(freq, type, duration, startTime = 0, gainLevel = 0.25) {
    if (isMuted) return;
    initAudioContext();
    if (!audioCtx) return;

    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime + startTime);

      gain.gain.setValueAtTime(gainLevel, audioCtx.currentTime + startTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + startTime + duration);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start(audioCtx.currentTime + startTime);
      osc.stop(audioCtx.currentTime + startTime + duration);
    } catch (e) {
      console.warn('[AUDIO TONE ERROR]', e);
    }
  }

  // 1. GREEN CHIME: Welcoming, pleasant 2-tone elevator/airport chime (C5 -> G5)
  function playGreenChime() {
    if (isMuted) return;
    playTone(523.25, 'sine', 0.45, 0.00, 0.35); // C5
    playTone(783.99, 'sine', 0.80, 0.16, 0.40); // G5
  }

  // 2. SAFETY ALARM: Urgent 3-pulse buzzer for hand near moving robot
  function playSafetyAlarm() {
    if (isMuted) return;
    const now = Date.now();
    if (now - lastAlarmSoundTime < 800) return; // Prevent excessive overlap
    lastAlarmSoundTime = now;

    [0, 0.14, 0.28].forEach((delay) => {
      playTone(880, 'sawtooth', 0.08, delay, 0.35);
    });
  }

  // 4. TAP CONFIRMATION PING
  function playTapPing() {
    playTone(659.25, 'sine', 0.35, 0.0, 0.25);
  }

  // Update UI Sound Button
  function updateSoundButton() {
    btnSoundToggle.textContent = isMuted ? '🔇' : '🔊';
  }
  updateSoundButton();

  btnSoundToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    isMuted = !isMuted;
    localStorage.setItem('xo_lights_muted', isMuted ? 'true' : 'false');
    updateSoundButton();
    if (!isMuted) {
      playTapPing();
    }
  });

  // Fullscreen Toggle
  btnFullscreenToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  // Screen Wake Lock API (keeps mobile/tablet screen awake on table)
  async function requestWakeLock() {
    if ('wakeLock' in navigator) {
      try {
        wakeLock = await navigator.wakeLock.request('screen');
        wakeLock.addEventListener('release', () => {
          wakeLock = null;
        });
      } catch (err) {
        console.warn('[WAKE LOCK FAILED]', err);
      }
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      requestWakeLock();
    }
  });

  // Start Overlay Click -> Activate Audio, WakeLock, Fullscreen
  startOverlay.addEventListener('click', () => {
    initAudioContext();
    requestWakeLock();
    startOverlay.classList.add('hidden');
    playTapPing();

    if (!document.fullscreenElement && window.innerWidth < 1024) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  });

  // Signal State Renderer
  function applySignalState(newState, titleText, subText, turnInfo, motionInfo) {
    const isStateChange = (lastSignalState !== newState);

    // Remove active classes
    ambientGlow.className = 'ambient-glow';
    lightRed.classList.remove('lit', 'alarm-flash');
    lightGreen.classList.remove('lit');
    instructionCard.className = 'instruction-card';

    // Apply new classes
    switch (newState) {
      case 'GREEN':
        ambientGlow.classList.add('green-aura');
        lightGreen.classList.add('lit');
        instructionCard.classList.add('state-green');
        badgeTurn.className = 'status-badge badge-green';
        badgeMotion.className = 'status-badge badge-green';
        if (isStateChange) {
          playGreenChime();
        }
        break;

      case 'RED':
        ambientGlow.classList.add('red-aura');
        lightRed.classList.add('lit');
        instructionCard.classList.add('state-red');
        badgeTurn.className = 'status-badge badge-red';
        badgeMotion.className = 'status-badge badge-red';
        // User requested: No sound on Red light
        break;

      case 'ALARM':
        ambientGlow.classList.add('alarm-aura');
        lightRed.classList.add('lit', 'alarm-flash');
        instructionCard.classList.add('state-alarm');
        badgeTurn.className = 'status-badge badge-red';
        badgeMotion.className = 'status-badge badge-red';
        playSafetyAlarm();
        break;

      case 'OFF':
      default:
        instructionCard.classList.add('state-off');
        badgeTurn.className = 'status-badge badge-off';
        badgeMotion.className = 'status-badge badge-off';
        break;
    }

    instructionMain.textContent = titleText;
    instructionSub.textContent = subText;
    badgeTurn.textContent = turnInfo;
    badgeMotion.textContent = motionInfo;

    lastSignalState = newState;
  }

  // Process incoming game state from WebSocket
  function processGameState(state) {
    if (!state) return;

    // 1. GAME NOT ACTIVE (Standby Menu / Waiting for Start)
    if (!state.game_active) {
      applySignalState(
        'OFF',
        'STANDBY',
        'Game not active. Waiting for start...',
        'NONE',
        'PARKED'
      );
      return;
    }

    // 2. GAME OVER
    if (state.game_over) {
      const winner = state.game_winner || 'DRAW';
      let winMsg = 'GAME OVER';
      if (winner === 'PLAYER') winMsg = `🏆 PLAYER WINS (${state.player_char})!`;
      else if (winner === 'ROBOT') winMsg = `🤖 ROBOT WINS (${state.robot_char})!`;
      else winMsg = '🤝 DRAW!';

      applySignalState(
        'OFF',
        'GAME OVER',
        winMsg,
        'FINISHED',
        'PARKED'
      );
      return;
    }

    // 3. ROBOT MOVING OR DRAWING
    const robotMoving = Boolean(state.robot_turn_in_progress);
    const handDetected = Boolean(state.hand_detected || state.hand_hazard);
    const hasPreexisting = Boolean(state.preexisting_marks_detected);
    const isPlayerGrid = (state.grid_drawer === 'PLAYER');
    const isGridLocked = Boolean(state.grid_locked);
    const isPlayerTurn = (state.current_turn === 'PLAYER');

    if (robotMoving) {
      // Safety violation: Hand detected while robot is actively in motion (triggered after 2 consecutive frames)
      if (handDetected) {
        motionHandFrames++;
      } else {
        motionHandFrames = 0;
      }

      const isAlarm = Boolean(state.hand_hazard || motionHandFrames >= 2);

      if (isAlarm) {
        applySignalState(
          'ALARM',
          '⚠️ STEP BACK!',
          'Hand detected near board while robot is moving! Do not touch!',
          'ROBOT',
          'MOVING'
        );
      } else {
        applySignalState(
          'RED',
          'STAND CLEAR',
          'Robot is physically moving / drawing. Do not approach the board.',
          'ROBOT',
          'MOVING'
        );
      }
      return;
    } else {
      motionHandFrames = 0;
    }

    // 4. PLAYER DRAWS GRID AND GRID IS NOT YET LOCKED
    // Robot is stationary at home waiting for the human to draw the 3x3 grid!
    if (isPlayerGrid && !isGridLocked) {
      applySignalState(
        'GREEN',
        'SAFE TO DRAW',
        'Draw a 3x3 grid on the whiteboard.',
        `PLAYER (${state.player_char})`,
        'PARKED'
      );
      return;
    }

    // 5. PRE-EXISTING MARKS WARNING (Player must erase marks before robot starts)
    if (hasPreexisting) {
      applySignalState(
        'GREEN',
        'SAFE TO APPROACH',
        'Please erase all marks from the grid so the robot can start.',
        'PLAYER',
        'PARKED'
      );
      return;
    }

    // 6. ROBOT IS STATIONARY, BUT IT IS ROBOT'S TURN AND GRID IS LOCKED
    // Robot is preparing next move on a locked grid... Stay back.
    if (!isPlayerTurn) {
      applySignalState(
        'RED',
        'STAND CLEAR',
        'Robot is preparing next move... Stay back.',
        'ROBOT',
        'PREPARING'
      );
      return;
    }

    // 7. PLAYER'S TURN / SAFE TO DRAW
    let subDetail = `Player's turn (${state.player_char}). Draw your mark on the board.`;
    if (state.invalid_cells && state.invalid_cells.length > 0) {
      subDetail = 'Erase the wrong mark with an eraser.';
    } else if (state.multiple_moves_detected) {
      subDetail = 'Erase extra marks so only 1 mark remains.';
    }

    applySignalState(
      'GREEN',
      'SAFE TO DRAW',
      subDetail,
      `PLAYER (${state.player_char})`,
      'PARKED'
    );
  }

  // WebSocket Connection Management
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      connDot.classList.add('online');
      connText.textContent = 'CONNECTED';
      // Immediate state check upon connection
      fetch('/api/state')
        .then(r => r.json())
        .then(data => processGameState(data))
        .catch(() => {});
    };

    ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        processGameState(state);
      } catch (err) {
        console.error('[WS PARSE ERROR]', err);
      }
    };

    ws.onclose = () => {
      connDot.classList.remove('online');
      connText.textContent = 'RECONNECTING...';
      applySignalState('OFF', 'RECONNECTING', 'Lost connection to robot server. Retrying...', 'OFFLINE', 'UNKNOWN');
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => {
      if (ws) ws.close();
    };
  }

  connectWebSocket();

  // Initial immediate fetch
  fetch('/api/state')
    .then(r => r.json())
    .then(data => processGameState(data))
    .catch(() => {});
});

