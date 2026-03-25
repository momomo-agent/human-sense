/**
 * AudioSenseEngine — speech recognition + volume monitoring
 *
 * Uses Web Speech API (webkitSpeechRecognition) for continuous speech-to-text
 * and Web Audio API (AnalyserNode) for real-time volume levels.
 */
export class AudioSenseEngine {
  constructor(options = {}) {
    this.wakeWords = (options.wakeWords || ['你好', 'hello', 'hey momo', 'momo'])
      .map(w => w.toLowerCase())
    this.lang = options.lang || 'zh-CN'

    // Callbacks
    this.onResult = null      // (text, isFinal, wakeWordDetected, wakeWord)
    this.onVolumeChange = null // (volume 0-1)
    this.onWake = null         // (wakeWord, fullText)

    this.recognition = null
    this.audioCtx = null
    this.analyser = null
    this.volumeRAF = null
    this._stopped = false
    this._supported = true
  }

  get supported() { return this._supported }

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
      const lower = text.toLowerCase()

      // Wake word detection
      let wakeDetected = false
      let matchedWakeWord = null
      for (const w of this.wakeWords) {
        if (lower.includes(w)) {
          wakeDetected = true
          matchedWakeWord = w
          break
        }
      }

      if (this.onResult) {
        this.onResult(text, isFinal, wakeDetected, matchedWakeWord)
      }

      if (wakeDetected && this.onWake) {
        this.onWake(matchedWakeWord, text)
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
