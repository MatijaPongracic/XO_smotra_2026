/**
 * sound.js - Web Audio API sintisajzer zvučnih efekata za digitalni semafor.
 * 100% radi offline, bez potrebe za vanjskim audio datotekama!
 */

class SoundManager {
  constructor() {
    this.ctx = null;
    this.muted = localStorage.getItem('xo_sound_muted') === 'true';
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  isMuted() {
    return this.muted;
  }

  setMuted(mute) {
    this.muted = mute;
    localStorage.setItem('xo_sound_muted', mute ? 'true' : 'false');
  }

  toggleMute() {
    this.setMuted(!this.muted);
    return this.muted;
  }

  // Pomoćna funkcija za sviranje jednostavnog sintetiziranog tona
  _playTone(freq, type, duration, startTime = 0, gainLevel = 0.2) {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, this.ctx.currentTime + startTime);

    gain.gain.setValueAtTime(gainLevel, this.ctx.currentTime + startTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + startTime + duration);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(this.ctx.currentTime + startTime);
    osc.stop(this.ctx.currentTime + startTime + duration);
  }

  // Zvuk za start igre ("LET'S PLAY!"): veseli 4-tonski arpeggio
  playGameStart() {
    if (this.muted) return;
    const notes = [261.63, 329.63, 392.00, 523.25]; // C4, E4, G4, C5
    notes.forEach((f, idx) => {
      this._playTone(f, 'sine', 0.25, idx * 0.08, 0.25);
    });
  }

  // Zvuk za zabilježeni potez (X ili O)
  playMove(isPlayer = true) {
    if (this.muted) return;
    const freq = isPlayer ? 660 : 440;
    this._playTone(freq, 'triangle', 0.12, 0, 0.2);
  }

  // Zvuk za pobjedu igrača: Trijumfalna fanfara
  playVictory() {
    if (this.muted) return;
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    notes.forEach((f, idx) => {
      this._playTone(f, 'triangle', 0.4, idx * 0.12, 0.3);
    });
  }

  // Zvuk za pobjedu robota
  playRobotWin() {
    if (this.muted) return;
    const notes = [440, 392, 349, 293]; // A4, G4, F4, D4
    notes.forEach((f, idx) => {
      this._playTone(f, 'sawtooth', 0.35, idx * 0.14, 0.18);
    });
  }

  // Zvuk za neriješeno
  playDraw() {
    if (this.muted) return;
    this._playTone(330, 'sine', 0.3, 0, 0.2);
    this._playTone(330, 'sine', 0.4, 0.25, 0.2);
  }

  // Zvuk za grešku / neispravan potez
  playWarning() {
    if (this.muted) return;
    this._playTone(220, 'square', 0.15, 0, 0.15);
    this._playTone(180, 'square', 0.25, 0.12, 0.15);
  }
}

// Globalna instanca
window.soundManager = new SoundManager();
