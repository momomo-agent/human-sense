/**
 * Human Sense — main entry
 * 
 * Architecture:
 * 1. Camera → MediaPipe Face Mesh (468 landmarks) + Hands
 * 2. Landmarks → Feature extractors (presence, attention, emotion, posture)
 * 3. Features → Synthesis engine (rule-based fusion)
 * 4. Synthesis → Dashboard UI
 * 
 * MediaPipe WASM files are self-hosted to avoid CDN issues in China.
 */

import { SenseEngine } from './engine.js'
import { Dashboard } from './dashboard.js'

const video = document.getElementById('camera')
const overlay = document.getElementById('overlay')
const statusEl = document.getElementById('status')

async function init() {
  try {
    // 1. Camera
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: 640, height: 480 }
    })
    video.srcObject = stream
    await video.play()

    overlay.width = video.videoWidth || 640
    overlay.height = video.videoHeight || 480

    statusEl.textContent = '加载模型中...'

    // 2. Engine
    const engine = new SenseEngine(video, overlay)
    await engine.init()

    // 3. Dashboard
    const dashboard = new Dashboard()

    statusEl.textContent = '感知中'
    statusEl.classList.add('active')

    // 4. Loop
    function loop() {
      const result = engine.detect()
      if (result) {
        dashboard.update(result)
      }
      requestAnimationFrame(loop)
    }

    requestAnimationFrame(loop)

  } catch (e) {
    console.error('Init failed:', e)
    statusEl.textContent = `错误: ${e.message}`
    statusEl.classList.add('error')
  }
}

init()
