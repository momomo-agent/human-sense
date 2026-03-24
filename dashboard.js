/**
 * Dashboard v3 — fullscreen HUD, null-safe
 */
export class Dashboard {
  constructor() {
    this.els = {}
    this.lastEventCount = 0
    this.active = false
    this.bindElements()
  }

  bindElements() {
    this.els = {
      faceCount: document.getElementById('face-count'),
      distanceBadge: document.getElementById('distance-pill'),
      gazeIndicator: document.getElementById('gaze-ring'),
      attentionBar: document.getElementById('attention-bar'),
      attentionPct: document.getElementById('attention-pct'),
      statusDot: document.getElementById('status-dot'),
      gaze: document.getElementById('s-gaze'),
      expression: document.getElementById('s-expression'),
      exprIcon: document.getElementById('s-expr-icon'),
      posture: document.getElementById('s-posture'),
      blink: document.getElementById('s-blink'),
      facing: document.getElementById('s-facing'),
      facingIcon: document.getElementById('s-facing-icon'),
      tilt: document.getElementById('s-tilt'),
      synthesisText: document.getElementById('synthesis-text'),
      timeline: document.getElementById('timeline-entries'),
      // New: hands & pose & objects
      handCount: document.getElementById('s-hand-count'),
      gesture: document.getElementById('s-gesture'),
      fingers: document.getElementById('s-fingers'),
      bodyPosture: document.getElementById('s-body-posture'),
      shoulders: document.getElementById('s-shoulders'),
      personRatio: document.getElementById('s-person-ratio'),
      objects: document.getElementById('s-objects'),
    }
  }

  // Safe setter — skip if element doesn't exist
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

    // Gaze indicator — position using screenX/screenY
    const gi = this.els.gazeIndicator
    if (gi && attention.gaze) {
      if (attention.gaze.screenX !== undefined) {
        gi.style.left = `${attention.gaze.screenX * 100}%`
        gi.style.top = `${attention.gaze.screenY * 100}%`
      }
      gi.classList.toggle('visible', presence.count > 0)
    }

    // Attention bar
    const bar = this.els.attentionBar
    const focusScore = attention.focus.score || 0
    if (bar) {
      bar.style.width = `${focusScore}%`
      bar.className = 'attention-fill'
      if (focusScore < 40) bar.classList.add('low')
      else if (focusScore < 70) bar.classList.add('medium')
    }
    this.setText('attentionPct', focusScore > 0 ? `${focusScore}%` : '—')

    // Sense pills
    this.setText('gaze', attention.gaze.region || '—')

    // Expression
    const exprText = emotion.expression || '—'
    const emojiMatch = exprText.match(/^([\p{Emoji}]+)\s*/u)
    if (emojiMatch) {
      this.setText('exprIcon', emojiMatch[1])
      this.setText('expression', exprText.slice(emojiMatch[0].length))
    } else {
      this.setText('expression', exprText)
    }

    this.setText('posture', emotion.posture || '—')
    this.setText('blink', attention.blinkRate > 0 ? `${attention.blinkRate}/min` : '—')

    this.setText('facing', presence.facing ? '是' : '否')
    this.setStyle('facing', 'color', presence.facing ? 'var(--green)' : 'var(--red)')
    this.setText('facingIcon', presence.facing ? '🎯' : '🚫')

    this.setText('tilt', emotion.tilt || '—')

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

    // ---- Hands ----
    const sense = result.sense
    if (sense) {
      this.setText('handCount', sense.handCount > 0 ? `${sense.handCount} ✋` : '0')

      if (sense.hands.length > 0) {
        const gestures = sense.hands
          .filter(h => h.gesture)
          .map(h => `${h.side === 'Left' ? '🫲' : '🫱'} ${h.gesture}`)
        this.setText('gesture', gestures.length > 0 ? gestures.join(' ') : '—')

        const fingerStr = sense.hands.map(h => {
          const f = h.fingers
          return [f.thumb ? '👍' : '·', f.index ? '☝' : '·', f.middle ? '🖕' : '·', f.ring ? '·' : '·', f.pinky ? '🤙' : '·'].join('')
        }).join(' | ')
        this.setText('fingers', fingerStr)
      } else {
        this.setText('gesture', '—')
        this.setText('fingers', '—')
      }

      // ---- Body ----
      if (sense.body) {
        this.setText('shoulders', sense.body.shoulderWidth ? `W:${sense.body.shoulderWidth}` : '—')
        this.setText('bodyPosture', sense.body.torsoLength ? `T:${sense.body.torsoLength}` : '—')
      } else {
        this.setText('shoulders', '—')
        this.setText('bodyPosture', '—')
      }

      // ---- Segmentation ----
      if (sense.segmentation) {
        this.setText('personRatio', `${(sense.segmentation.personRatio * 100).toFixed(0)}%`)
      } else {
        this.setText('personRatio', '—')
      }

      // ---- Objects ----
      if (sense.objects && sense.objects.length > 0) {
        const objStr = sense.objects.map(o => `${o.label}(${(o.confidence * 100).toFixed(0)}%)`).join(' ')
        this.setText('objects', objStr)
      } else {
        this.setText('objects', '—')
      }
    }
  }
}
