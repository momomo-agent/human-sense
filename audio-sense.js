/**
 * AudioSenseEngine — speech recognition + volume monitoring + wake word detection
 *
 * Uses Web Speech API (webkitSpeechRecognition) for continuous speech-to-text
 * and Web Audio API (AnalyserNode) for real-time volume levels.
 *
 * Wake word detection uses multi-modal fusion:
 * - With camera: silence gap + facing screen + wake word position
 * - Without camera: silence gap + wake word at sentence start
 */
export class AudioSenseEngine {
  constructor(options = {}) {
    this.wakeWords = (options.wakeWords || ['你好', 'hello', 'hey momo', 'momo'])
      .map(w => w.toLowerCase())
    this.lang = options.lang || 'zh-CN'

    // Callbacks
    this.onResult = null      // (text, isFinal, wakeJudgment)
    this.onVolumeChange = null // (volume 0-1)
    this.onWake = null         // (wakeWord, fullText, judgment)

    this.recognition = null
    this.audioCtx = null
    this.analyser = null
    this.volumeRAF = null
    this._stopped = false
    this._supported = true

    // Wake judgment state
    this._lastSpeechTime = 0      // timestamp of last speech activity
    this._silenceThresholdMs = 1500 // silence gap to count as "new utterance"
    this._facing = null            // external signal: is user facing screen?
    this._hasCamera = false        // whether camera data is available
  }

  get supported() { return this._supported }

  /** Update visual context from camera (call from main loop) */
  updateVisualContext(facing) {
    this._facing = facing
    this._hasCamera = true
  }

  /** Clear camera context (no camera available) */
  clearVisualContext() {
    this._facing = null
    this._hasCamera = false
  }

  async start() {
    this._stopped = false

    // ---- Speech Recognition ----
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      this._supported = false
      console.warn('SpeechRecognition not supported')
      return false
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = this.lang
    recognition.maxAlternatives = 1

    recognition.onresult = (event) => {
      // Get the latest result
      let interimText = ''
      let finalText = ''
      let isFinal = false

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalText += transcript
          isFinal = true
        } else {
          interimText += transcript
        }
      }

      const text = isFinal ? finalText : interimText
      const lower = text.toLowerCase().trim()

      // Wake word detection with multi-modal fusion
      const judgment = this._judgeWake(lower, text, isFinal)

      // Track speech activity timing
      if (text.length > 0) {
        this._lastSpeechTime = Date.now()
      }

      if (this.onResult) {
        this.onResult(text, isFinal, judgment)
      }

      if (judgment.isWake && isFinal && this.onWake) {
        this.onWake(judgment.wakeWord, text, judgment)
      }
      }
    }

    recognition.onerror = (event) => {
      // 'no-speech' and 'aborted' are normal — auto-restart handles them
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.warn('SpeechRecognition error:', event.error)
      }
    }

    recognition.onend = () => {
      // Auto-restart unless manually stopped
      if (!this._stopped) {
        try { recognition.start() } catch (e) { /* already started */ }
      }
    }

    this.recognition = recognition

    try {
      recognition.start()
    } catch (e) {
      console.warn('SpeechRecognition start failed:', e)
      return false
    }

    // ---- Volume monitoring via Web Audio API ----
    this._startVolumeMeter()

    return true
  }

  async _startVolumeMeter() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)()
      const source = this.audioCtx.createMediaStreamSource(stream)
      this.analyser = this.audioCtx.createAnalyser()
      this.analyser.fftSize = 256
      source.connect(this.analyser)

      const dataArray = new Uint8Array(this.analyser.frequencyBinCount)

      const tick = () => {
        if (this._stopped) return
        this.analyser.getByteFrequencyData(dataArray)

        // RMS volume normalized to 0-1
        let sum = 0
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i] * dataArray[i]
        }
        const rms = Math.sqrt(sum / dataArray.length) / 255

        if (this.onVolumeChange) {
          this.onVolumeChange(rms)
        }

        this.volumeRAF = requestAnimationFrame(tick)
      }
      this.volumeRAF = requestAnimationFrame(tick)
    } catch (e) {
      console.warn('Volume meter init failed:', e)
    }
  }

  /**
   * Multi-modal wake word judgment
   *
   * Returns: { isWake, wakeWord, confidence, reason, signals }
   *
   * With camera:  (silence + facing + position) → high confidence
   * Without camera: (silence + position) → medium confidence
   */
  _judgeWake(lower, originalText, isFinal) {
    const result = { isWake: false, wakeWord: null, confidence: 0, reason: '', signals: {} }

    // 1. Find wake word and its position
    let matchedWord = null
    let matchIdx = -1
    for (const w of this.wakeWords) {
      const idx = lower.indexOf(w)
      if (idx >= 0) {
        matchedWord = w
        matchIdx = idx
        break
      }
    }
    if (!matchedWord) return result

    result.wakeWord = matchedWord

    // 2. Compute signals
    const now = Date.now()
    const silenceGap = now - this._lastSpeechTime
    const isAfterSilence = this._lastSpeechTime === 0 || silenceGap > this._silenceThresholdMs
    const isAtStart = matchIdx <= 2  // allow 1-2 chars of whitespace/punctuation
    const isFacing = this._hasCamera ? this._facing === true : null

    result.signals = {
      silenceGap: isAfterSilence,
      atStart: isAtStart,
      facing: isFacing,
      hasCamera: this._hasCamera
    }

    // 3. Score
    let score = 0

    // Silence before speaking — strong signal (someone just started talking)
    if (isAfterSilence) score += 0.4

    // Wake word at start of utterance
    if (isAtStart) score += 0.35

    // Facing screen (camera available) — bonus signal, never penalty
    if (this._hasCamera && isFacing) {
      score += 0.2
    }

    result.confidence = Math.max(0, Math.min(1, score))

    // 4. Threshold
    if (result.confidence >= 0.5) {
      result.isWake = true
      // Build reason
      const reasons = []
      if (isAfterSilence) reasons.push('静默后开口')
      if (isAtStart) reasons.push('句首')
      if (isFacing === true) reasons.push('面朝屏幕')
      result.reason = reasons.join(' + ')
    } else {
      // Explain why not triggered
      const reasons = []
      if (!isAfterSilence) reasons.push('连续说话中')
      if (!isAtStart) reasons.push('唤醒词在句中')
      if (isFacing === false) reasons.push('没看屏幕')
      result.reason = reasons.join(' + ')
    }

    return result
  }

  stop() {
    this._stopped = true

    if (this.recognition) {
      try { this.recognition.stop() } catch (e) { /* ok */ }
      this.recognition = null
    }

    if (this.volumeRAF) {
      cancelAnimationFrame(this.volumeRAF)
      this.volumeRAF = null
    }

    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {})
      this.audioCtx = null
    }
  }
}
