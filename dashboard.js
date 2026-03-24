/**
 * Dashboard — renders sense data to DOM
 */
export class Dashboard {
  constructor() {
    this.els = {
      count: document.getElementById('m-count'),
      distance: document.getElementById('m-distance'),
      facing: document.getElementById('m-facing'),
      gaze: document.getElementById('m-gaze'),
      blink: document.getElementById('m-blink'),
      focus: document.getElementById('m-focus'),
      expression: document.getElementById('m-expression'),
      posture: document.getElementById('m-posture'),
      tilt: document.getElementById('m-tilt'),
      synthesis: document.getElementById('m-synthesis'),
      timeline: document.getElementById('timeline-entries'),
    }
    this.lastEventCount = 0
  }

  update(result) {
    const { presence, attention, emotion, synthesis } = result

    // Presence
    this.els.count.textContent = presence.count
    this.els.distance.textContent = presence.count > 0 ? `${presence.distance}x` : '-'
    this.els.facing.textContent = presence.count > 0 ? (presence.facing ? '✓ 是' : '✗ 否') : '-'
    this.els.facing.style.color = presence.facing ? '#4ade80' : '#f87171'

    // Attention
    this.els.gaze.textContent = attention.gaze.region
    this.els.blink.textContent = attention.blinkRate > 0 ? `${attention.blinkRate}/min` : '-'
    this.els.focus.textContent = attention.focus.level !== '-' ? 
      `${attention.focus.level} ${attention.focus.score}%` : '-'
    
    // Color code focus
    const focusColors = { '高': '#4ade80', '中': '#fbbf24', '低': '#f87171' }
    this.els.focus.style.color = focusColors[attention.focus.level] || '#e0e0e0'

    // Emotion
    this.els.expression.textContent = emotion.expression
    this.els.posture.textContent = emotion.posture
    this.els.tilt.textContent = emotion.tilt

    // Synthesis
    this.els.synthesis.textContent = synthesis.text

    // Timeline
    if (synthesis.events.length !== this.lastEventCount) {
      this.lastEventCount = synthesis.events.length
      this.els.timeline.innerHTML = synthesis.events
        .slice(-20)
        .reverse()
        .map(e => `<div class="timeline-entry">
          <span class="time">${e.time}</span>
          <span class="event">${e.text}</span>
        </div>`)
        .join('')
    }
  }
}
