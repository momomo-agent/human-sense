import { SenseEngine } from './engine.js'
import { Dashboard } from './dashboard.js'

const video = document.getElementById('camera')
const overlay = document.getElementById('overlay')
const statusDot = document.getElementById('status-dot')

async function init() {
  const startBtn = document.getElementById('start-btn')
  const startOverlay = document.getElementById('start-overlay')
  const app = document.getElementById('app')

  await new Promise(resolve => {
    startBtn.addEventListener('click', resolve, { once: true })
  })
  startOverlay.style.display = 'none'
  app.style.display = 'grid'

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: 640, height: 480 }
    })
    video.srcObject = stream
    await video.play()

    overlay.width = video.videoWidth || 640
    overlay.height = video.videoHeight || 480

    const engine = new SenseEngine(video, overlay)
    await engine.init()

    const dashboard = new Dashboard()

    function loop() {
      const result = engine.detect()
      if (result) dashboard.update(result)
      requestAnimationFrame(loop)
    }

    requestAnimationFrame(loop)

  } catch (e) {
    console.error('Init failed:', e)
    document.getElementById('synthesis-text').textContent = `错误: ${e.message}`
    statusDot.style.background = 'var(--red)'
  }
}

init()
