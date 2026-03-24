/**
 * Dashboard v2 — visual-first, card-based
 */
export class Dashboard {
  constructor() {
    this.els = {
      // Camera overlays
      faceCount: document.getElementById('face-count'),
      faceBadge: document.getElementById('face-badge'),
      distanceBadge: document.getElementById('distance-badge'),
      gazeIndicator: document.getElementById('gaze-ring'),
      attentionBar: document.getElementById('attention-bar'),
      attentionPct: document.getElementById('attention-pct'),
      statusDot: document.getElementById('status-dot'),

      // Sense cards
      gaze: document.getElementById('s-gaze'),
      expression: document.getElementById('s-expression'),
      exprIcon: document.getElementById('s-expr-icon'),
      posture: document.getElementById('s-posture'),
      blink: document.getElementById('s-blink'),
      facing: document.getElementById('s-facing'),
      facingIcon: document.getElementById('s-facing-icon'),
      tilt: document.getElementById('s-tilt'),

      // Synthesis
      synthesisText: document.getElementById('synthesis-text'),

      // Timeline
      timeline: document.getElementById('timeline-entries'),
    }
    this.lastEventCount = 0
    this.active = false
  }

  update(result) {
    if (!result) return
    const { presence, attention, emotion, synthesis } = result

    // Activate status
    if (!this.active) {
      this.active = true
      this.els.statusDot.classList.add('active')
    }

    // Face count
    this.els.faceCount.textContent = presence.count
    this.els.faceCount.style.color = presence.count > 0 ? 'var(--green)' : 'var(--red)'

    // Distance
    if (presence.count > 0 && presence.distance !== '-') {
      const d = parseFloat(presence.distance)
      let label = '近'
      if (d > 1.5) label = '远'
      else if (d > 0.8) label = '适中'
      this.els.distanceBadge.textContent = `📏 ${label}`
    } else {
      this.els.distanceBadge.textContent = '—'
    }

    // Gaze indicator position on camera
    if (attention.gaze && attention.gaze.x !== undefined) {
      const gx = 50 - attention.gaze.x * 40
      const gy = 50 + attention.gaze.y * 40
      this.els.gazeIndicator.style.left = `${Math.max(10, Math.min(90, gx))}%`
      this.els.gazeIndicator.style.top = `${Math.max(10, Math.min(90, gy))}%`
      this.els.gazeIndicator.classList.toggle('visible', presence.count > 0)
    }

    // Attention bar
    const focusScore = attention.focus.score || 0
    this.els.attentionBar.style.width = `${focusScore}%`
    this.els.attentionBar.className = 'attention-bar'
    if (focusScore < 40) this.els.attentionBar.classList.add('low')
    else if (focusScore < 70) this.els.attentionBar.classList.add('medium')
    this.els.attentionPct.textContent = focusScore > 0 ? `${focusScore}%` : '—'

    // Sense cards
    this.els.gaze.textContent = attention.gaze.region || '—'
    
    // Expression — extract emoji and text separately
    const exprText = emotion.expression || '—'
    const emojiMatch = exprText.match(/^([\p{Emoji}]+)\s*/u)
    if (emojiMatch) {
      this.els.exprIcon.textContent = emojiMatch[1]
      this.els.expression.textContent = exprText.slice(emojiMatch[0].length)
    } else {
      this.els.expression.textContent = exprText
    }

    this.els.posture.textContent = emotion.posture || '—'
    this.els.blink.textContent = attention.blinkRate > 0 ? `${attention.blinkRate}/min` : '—'
    
    this.els.facing.textContent = presence.facing ? '是' : '否'
    this.els.facing.style.color = presence.facing ? 'var(--green)' : 'var(--red)'
    this.els.facingIcon.textContent = presence.facing ? '🎯' : '🚫'

    this.els.tilt.textContent = emotion.tilt || '—'

    // Synthesis
    this.els.synthesisText.textContent = synthesis.text

    // Timeline
    if (synthesis.events.length !== this.lastEventCount) {
      this.lastEventCount = synthesis.events.length
      this.els.timeline.innerHTML = synthesis.events
        .slice(-15)
        .reverse()
        .map(e => `<div class="timeline-entry">
          <span class="time">${e.time}</span>
          <span class="event">${e.text}</span>
        </div>`)
        .join('')
    }
  }
}
