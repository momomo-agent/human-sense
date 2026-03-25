/**
 * Dashboard v4 — card-panel HUD with ring chart + finger dots
 */
export class Dashboard {
  constructor() {
    this.els = {}
    this.lastEventCount = 0
    this.active = false
    this.focusCircumference = 2 * Math.PI * 34 // r=34 in SVG
    this.bindElements()
  }

  bindElements() {
    this.els = {
      faceCount: document.getElementById('face-count'),
      distanceBadge: document.getElementById('distance-pill'),
      gazeIndicator: document.getElementById('gaze-ring'),
      statusDot: document.getElementById('status-dot'),

      // Face section
      exprIcon: document.getElementById('s-expr-icon'),
      expression: document.getElementById('s-expression'),
      facingIcon: document.getElementById('s-facing-icon'),
      facing: document.getElementById('s-facing'),
      posture: document.getElementById('s-posture'),
      tilt: document.getElementById('s-tilt'),

      // Attention section
      focusRingFill: document.getElementById('focus-ring-fill'),
      attentionPct: document.getElementById('attention-pct'),
      gaze: document.getElementById('s-gaze'),
      blink: document.getElementById('s-blink'),

      // Hands section
      gestureEmoji: document.getElementById('s-gesture-emoji'),
      gesture: document.getElementById('s-gesture'),
      handLeft: document.getElementById('hand-left'),
      handRight: document.getElementById('hand-right'),
      fingersLeft: document.getElementById('fingers-left'),
      fingersRight: document.getElementById('fingers-right'),
      action: document.getElementById('s-action'),

      // Body section
      bodyPosture: document.getElementById('s-body-posture'),
      shoulderBar: document.getElementById('shoulder-bar'),
      torsoBar: document.getElementById('torso-bar'),
      personRatio: document.getElementById('s-person-ratio'),

      // Synthesis + timeline
      synthesisText: document.getElementById('synthesis-text'),
      timeline: document.getElementById('timeline-entries'),

      // Audio section
      volumeBar: document.getElementById('volume-bar'),
      modelStatus: document.getElementById('s-model-status'),
      audioStatus: document.getElementById('s-audio-status'),
      wakeStatus: document.getElementById('s-wake-status'),
    }
  }

  setText(key, val) {
    const el = this.els[key]
    if (el) el.textContent = val
  }

  setStyle(key, prop, val) {
    const el = this.els[key]
    if (el) el.style[prop] = val
  }

  update(result) {
    if (!result) return
    const { presence, attention, emotion, synthesis } = result

    // Activate status
    if (!this.active && this.els.statusDot) {
      this.active = true
      this.els.statusDot.classList.add('active')
    }

    // Face count
    this.setText('faceCount', presence.count)
    this.setStyle('faceCount', 'color', presence.count > 0 ? 'var(--green)' : 'var(--red)')

    // Distance
    if (presence.count > 0 && presence.distance !== '-') {
      const d = parseFloat(presence.distance)
      let label = '近'
      if (d > 1.5) label = '远'
      else if (d > 0.8) label = '适中'
      this.setText('distanceBadge', `📏 ${label}`)
    } else {
      this.setText('distanceBadge', '—')
    }

    // Gaze dot
    const gi = this.els.gazeIndicator
    if (gi && attention.gaze) {
      if (attention.gaze.screenX !== undefined) {
        gi.style.left = `${attention.gaze.screenX * 100}%`
        gi.style.top = `${attention.gaze.screenY * 100}%`
      }
      gi.classList.toggle('visible', presence.count > 0)
    }

    // ---- Face section ----
    const exprText = emotion.expression || '—'
    const emojiMatch = exprText.match(/^([\p{Emoji}]+)\s*/u)
    if (emojiMatch) {
      this.setText('exprIcon', emojiMatch[1])
      this.setText('expression', exprText.slice(emojiMatch[0].length))
    } else {
      this.setText('exprIcon', '😐')
      this.setText('expression', exprText)
    }

    this.setText('facing', presence.facing ? '面对' : '未面对')
    this.setStyle('facing', 'color', presence.facing ? 'var(--green)' : 'var(--red)')
    this.setText('facingIcon', presence.facing ? '🎯' : '🚫')
    this.setText('posture', emotion.posture || '—')
    this.setText('tilt', emotion.tilt || '—')

    // ---- Attention section (ring chart) ----
    const focusScore = attention.focus.score || 0
    const ring = this.els.focusRingFill
    if (ring) {
      const offset = this.focusCircumference * (1 - focusScore / 100)
      ring.style.strokeDashoffset = offset
      ring.className.baseVal = 'focus-ring-fill'
      if (focusScore < 35) ring.classList.add('low')
      else if (focusScore < 65) ring.classList.add('medium')
    }
    this.setText('attentionPct', focusScore > 0 ? `${focusScore}%` : '—')
    this.setText('gaze', attention.gaze.region || '—')
    this.setText('blink', attention.blinkRate > 0 ? `${attention.blinkRate}/min` : '—')

    // ---- Hands section ----
    const sense = result.sense
    const customGestures = result.customGestures || []
    const actions = result.actions || []

    if (sense) {
      // Reset hand indicators
      if (this.els.handLeft) this.els.handLeft.classList.remove('active')
      if (this.els.handRight) this.els.handRight.classList.remove('active')

      // Determine gesture to show (MediaPipe > Custom)
      let gestureEmoji = '—'
      let gestureLabel = '—'

      if (sense.hands.length > 0) {
        // Find best gesture from MediaPipe hands
        for (const h of sense.hands) {
          if (h.gesture) {
            gestureLabel = h.gesture
            gestureEmoji = this.gestureToEmoji(h.gesture)
          }
        }

        // Custom gestures override if no MediaPipe gesture
        if (gestureLabel === '—' && customGestures.length > 0) {
          const cg = customGestures[0]
          gestureLabel = cg.name
          gestureEmoji = cg.emoji || '✋'
        }

        // Update finger dots for each hand
        for (const h of sense.hands) {
          const isLeft = h.side === 'Left'
          const indicator = isLeft ? this.els.handLeft : this.els.handRight
          const dots = isLeft ? this.els.fingersLeft : this.els.fingersRight
          if (indicator) indicator.classList.add('active')
          if (dots) {
            const fingers = [h.fingers.thumb, h.fingers.index, h.fingers.middle, h.fingers.ring, h.fingers.pinky]
            const dotEls = dots.querySelectorAll('.fdot')
            for (let i = 0; i < dotEls.length; i++) {
              dotEls[i].classList.toggle('on', fingers[i] || false)
            }
          }
        }
      }

      this.setText('gestureEmoji', gestureEmoji)
      this.setText('gesture', gestureLabel)

      // Actions
      if (actions.length > 0) {
        const actionText = actions.map(a => `${a.emoji} ${a.name}`).join(' ')
        this.setText('action', actionText)
        if (this.els.action) this.els.action.classList.add('active')
      } else {
        this.setText('action', '—')
        if (this.els.action) this.els.action.classList.remove('active')
      }

      // ---- Body section ----
      if (sense.body) {
        // Posture from head pose
        const posture = emotion.posture || '—'
        const tilt = emotion.tilt || '正'
        this.setText('bodyPosture', `${posture} · ${tilt}`)

        // Shoulder bar (normalized: 0.1-0.5 typical range)
        if (sense.body.shoulderWidth && this.els.shoulderBar) {
          const pct = Math.min(100, Math.max(0, (sense.body.shoulderWidth / 0.5) * 100))
          this.els.shoulderBar.style.width = `${pct}%`
        }

        // Torso bar
        if (sense.body.torsoLength && this.els.torsoBar) {
          const pct = Math.min(100, Math.max(0, (sense.body.torsoLength / 0.5) * 100))
          this.els.torsoBar.style.width = `${pct}%`
        }
      } else {
        this.setText('bodyPosture', '—')
      }

      // Person ratio
      if (sense.segmentation) {
        this.setText('personRatio', `${(sense.segmentation.personRatio * 100).toFixed(0)}%`)
      } else {
        this.setText('personRatio', '—')
      }
    }

    // Synthesis
    this.setText('synthesisText', synthesis.text)

    // Timeline
    const tl = this.els.timeline
    if (tl && synthesis.events.length !== this.lastEventCount) {
      this.lastEventCount = synthesis.events.length
      tl.innerHTML = synthesis.events
        .slice(-15)
        .reverse()
        .map(e => `<div class="timeline-entry">
          <span class="time">${e.time}</span>
          <span class="event">${e.text}</span>
        </div>`)
        .join('')
    }
  }

  updateAudio({ text, isFinal, wakeDetected, wakeWord, confidence, reason }) {
    const el = this.els.audioStatus
    if (el) {
      if (wakeDetected) {
        el.textContent = '已唤醒'
        el.className = 'detail-val woke'
      } else if (text) {
        el.textContent = '识别中'
        el.className = 'detail-val listening'
      } else {
        el.textContent = '监听中'
        el.className = 'detail-val listening'
      }
    }

    if (wakeDetected && wakeWord && this.els.wakeStatus) {
      const pct = confidence ? `${(confidence * 100).toFixed(0)}%` : ''
      const reasonText = reason || ''
      this.els.wakeStatus.textContent = `${wakeWord} ${pct}`
      this.els.wakeStatus.title = reasonText  // hover to see reason
      this.els.wakeStatus.classList.add('active')
    }
  }

  updateVolume(vol) {
    if (this.els.volumeBar) {
      const pct = Math.min(100, Math.max(0, vol * 100 * 2.5)) // amplify for visibility
      this.els.volumeBar.style.height = `${pct}%`
    }

    // Set status to listening if not already woke
    const el = this.els.audioStatus
    if (el && !el.classList.contains('woke')) {
      el.textContent = '监听中'
      el.className = 'detail-val listening'
    }
  }

  updateModelStatus(status, message) {
    const el = this.els.modelStatus
    if (!el) return
    el.textContent = message
    el.className = 'detail-val'
    if (status === 'loading') el.classList.add('model-loading')
    else if (status === 'ready') el.classList.add('model-ready')
    else if (status === 'error') el.classList.add('model-error')
  }

  gestureToEmoji(name) {
    const map = {
      'Closed_Fist': '✊',
      'Open_Palm': '🖐️',
      'Pointing_Up': '☝️',
      'Thumb_Down': '👎',
      'Thumb_Up': '👍',
      'Victory': '✌️',
      'ILoveYou': '🤟',
    }
    return map[name] || '✋'
  }
}
