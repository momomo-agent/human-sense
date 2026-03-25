/**
 * AudioSenseEngine — speech recognition + volume monitoring + wake word detection
 *
 * Uses Whisper WASM (Transformers.js) in a Web Worker for local speech-to-text
 * and Web Audio API (ScriptProcessorNode) for real-time volume + VAD.
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
    this.onModelStatus = null  // (status, message) model loading status

    this.worker = null
    this.audioCtx = null
    this._stopped = false
    this._supported = true
    this._modelReady = false

    // Audio capture state
    this.audioChunks = []       // Float32Array buffers
    this.silenceStart = 0
    this.isSpeaking = false
    this.vadThreshold = 0.01    // RMS threshold for VAD
    this.chunkDurationMs = 3000 // send to whisper every 3s while speaking
    this.lastChunkTime = 0

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

    // 1. Start Worker and load Whisper model
    this.worker = new Worker('./whisper-worker.js', { type: 'module' })
    this.worker.onmessage = (e) => this._handleWorkerMessage(e)
    this.worker.postMessage({ type: 'init' })

    // 2. Get microphone and set up audio capture
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
      const source = this.audioCtx.createMediaStreamSource(stream)

      // ScriptProcessorNode for PCM capture + VAD
      const processor = this.audioCtx.createScriptProcessor(4096, 1, 1)
      source.connect(processor)
      processor.connect(this.audioCtx.destination)

      processor.onaudioprocess = (e) => {
        if (this._stopped) return

        const data = e.inputBuffer.getChannelData(0)
        const rms = this._computeRMS(data)

        if (this.onVolumeChange) this.onVolumeChange(rms)

        // VAD
        if (rms > this.vadThreshold) {
          if (!this.isSpeaking) {
            this.isSpeaking = true
            this.audioChunks = []
            this.lastChunkTime = Date.now()
          }
          this.silenceStart = 0
          this.audioChunks.push(new Float32Array(data))
        } else if (this.isSpeaking) {
          if (!this.silenceStart) this.silenceStart = Date.now()
          this.audioChunks.push(new Float32Array(data))

          // Silence > 800ms means utterance ended
          if (Date.now() - this.silenceStart > 800) {
            this.isSpeaking = false
            this._sendToWhisper(true)
          }
        }

        // Send interim results every chunkDurationMs while speaking
        if (this.isSpeaking && Date.now() - this.lastChunkTime > this.chunkDurationMs) {
          this._sendToWhisper(false)
        }
      }

      this._stream = stream
      this._processor = processor
      this._source = source
    } catch (e) {
      console.warn('Audio capture init failed:', e)
      this._supported = false
      return false
    }

    return true
  }

  _computeRMS(data) {
    let sum = 0
    for (let i = 0; i < data.length; i++) {
      sum += data[i] * data[i]
    }
    return Math.sqrt(sum / data.length)
  }

  _sendToWhisper(isFinal) {
    if (!this._modelReady || this.audioChunks.length === 0) return

    // Merge Float32Array chunks
    const totalLength = this.audioChunks.reduce((sum, c) => sum + c.length, 0)
    const merged = new Float32Array(totalLength)
    let offset = 0
    for (const chunk of this.audioChunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }

    this.worker.postMessage({ type: 'transcribe', audio: merged, isFinal }, [merged.buffer])

    if (isFinal) {
      this.audioChunks = []
    }
    this.lastChunkTime = Date.now()
  }

  _handleWorkerMessage(e) {
    const { type, status, message, text } = e.data

    if (type === 'status') {
      this._modelReady = (status === 'ready')
      if (this.onModelStatus) this.onModelStatus(status, message)
    }

    if (type === 'result') {
      const trimmed = (text || '').trim()
      if (!trimmed) return

      const lower = trimmed.toLowerCase()

      // Wake word detection with multi-modal fusion
      const judgment = this._judgeWake(lower, trimmed, true)

      // Track speech activity timing
      this._lastSpeechTime = Date.now()

      if (this.onResult) {
        this.onResult(trimmed, true, judgment)
      }

      if (judgment.isWake && this.onWake) {
        this.onWake(judgment.wakeWord, trimmed, judgment)
      }
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

    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }

    if (this._processor) {
      this._processor.disconnect()
      this._processor = null
    }

    if (this._source) {
      this._source.disconnect()
      this._source = null
    }

    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop())
      this._stream = null
    }

    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {})
      this.audioCtx = null
    }
  }
}
