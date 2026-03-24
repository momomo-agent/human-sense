import { extractFrame, IDX } from './sense-data.js'

/**
 * SenseEngine — face landmark detection + feature extraction
 * 
 * Uses @mediapipe/tasks-vision FaceLandmarker (self-hosted WASM + model)
 * Fallback: Chrome FaceDetector API (limited features)
 * 
 * Pipeline: Camera → FaceLandmarker (478 landmarks incl. iris) → Feature extractors → Synthesis
 */

// ---- EMA utility ----
class EMA {
  constructor(alpha = 0.3) {
    this.alpha = alpha
    this.value = null
  }
  update(v) {
    if (this.value === null) { this.value = v; return v }
    this.value = this.value * (1 - this.alpha) + v * this.alpha
    return this.value
  }
  get() { return this.value }
}

// ---- Blink detector ----
class BlinkDetector {
  constructor() {
    this.history = []
    this.wasOpen = true
    this.threshold = 0.2
  }

  computeEAR(eyeLandmarks) {
    if (!eyeLandmarks || eyeLandmarks.length < 6) return 1
    const [p1, p2, p3, p4, p5, p6] = eyeLandmarks
    const vertical1 = Math.hypot(p2.x - p6.x, p2.y - p6.y)
    const vertical2 = Math.hypot(p3.x - p5.x, p3.y - p5.y)
    const horizontal = Math.hypot(p1.x - p4.x, p1.y - p4.y)
    return (vertical1 + vertical2) / (2 * horizontal + 0.001)
  }

  update(ear) {
    const isOpen = ear > this.threshold
    if (this.wasOpen && !isOpen) {
      this.history.push(Date.now())
      const cutoff = Date.now() - 60000
      this.history = this.history.filter(t => t > cutoff)
    }
    this.wasOpen = isOpen
    return this.getRate()
  }

  getRate() {
    const cutoff = Date.now() - 60000
    return this.history.filter(t => t > cutoff).length
  }
}

// ---- Expression classifier ----
class ExpressionClassifier {
  classify(landmarks) {
    if (!landmarks || landmarks.length < 468) return { expression: '未知', confidence: 0 }

    const upperLip = landmarks[13]
    const lowerLip = landmarks[14]
    const mouthLeft = landmarks[61]
    const mouthRight = landmarks[291]

    const mouthOpen = Math.abs(upperLip.y - lowerLip.y)
    const mouthWidth = Math.abs(mouthLeft.x - mouthRight.x)
    const mouthRatio = mouthOpen / (mouthWidth + 0.001)

    const leftBrow = landmarks[66]
    const leftEye = landmarks[159]
    const browEyeDist = Math.abs(leftBrow.y - leftEye.y)

    const mouthCenter = (upperLip.y + lowerLip.y) / 2
    const cornerAvgY = (mouthLeft.y + mouthRight.y) / 2
    const smileScore = mouthCenter - cornerAvgY

    if (mouthRatio > 0.4) return { expression: '😮 惊讶/说话', confidence: 0.7 }
    if (smileScore > 0.01) return { expression: '😊 微笑', confidence: 0.6 }
    if (browEyeDist > 0.045) return { expression: '🤔 思考/困惑', confidence: 0.5 }
    return { expression: '😐 平静', confidence: 0.5 }
  }
}

// ---- Head pose estimator ----
class HeadPoseEstimator {
  estimate(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return { yaw: 0, pitch: 0, roll: 0, facing: true, posture: '未知', tilt: '正' }
    }

    const nose = landmarks[1]
    const leftFace = landmarks[234]
    const rightFace = landmarks[454]
    const forehead = landmarks[10]
    const chin = landmarks[152]

    const faceCenter = (leftFace.x + rightFace.x) / 2
    const faceWidth = Math.abs(rightFace.x - leftFace.x)
    const yaw = (nose.x - faceCenter) / (faceWidth + 0.001)

    const faceMidY = (forehead.y + chin.y) / 2
    const faceHeight = Math.abs(chin.y - forehead.y)
    const pitch = (nose.y - faceMidY) / (faceHeight + 0.001)

    const leftEye = landmarks[33]
    const rightEye = landmarks[263]
    const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / Math.PI

    const facing = Math.abs(yaw) < 0.15 && Math.abs(pitch) < 0.2
    let posture = '正坐'
    if (pitch < -0.15) posture = '前倾'
    else if (pitch > 0.15) posture = '后仰'
    if (Math.abs(yaw) > 0.25) posture = '侧头'

    let tilt = '正'
    if (roll > 8) tilt = '右倾'
    else if (roll < -8) tilt = '左倾'

    return { yaw, pitch, roll, facing, posture, tilt }
  }
}

// ---- Gaze estimator ----
class GazeEstimator {
  constructor() {
    this.gazeXEMA = new EMA(0.25)
    this.gazeYEMA = new EMA(0.25)
    // Calibration: maps iris ratio to screen position
    // iris ratioX ~0.35 = looking right, ~0.65 = looking left (mirrored)
    // We map to -1 (left) to +1 (right) screen space
    this.scaleX = 3.5   // sensitivity
    this.scaleY = 4.0
  }

  estimate(landmarks) {
    if (!landmarks || landmarks.length < 478) return { region: '未知', looking: false, x: 0, y: 0, screenX: 0.5, screenY: 0.5 }

    // Iris position within eye socket
    const le33 = landmarks[33], le133 = landmarks[133], liris = landmarks[468]
    const leWidth = Math.abs(le133.x - le33.x)
    const lRatioX = (liris.x - le33.x) / (leWidth + 0.001)

    const re362 = landmarks[362], re263 = landmarks[263], riris = landmarks[473]
    const reWidth = Math.abs(re263.x - re362.x)
    const rRatioX = (riris.x - re362.x) / (reWidth + 0.001)

    // Average iris horizontal position (0.5 = center)
    const irisX = (lRatioX + rRatioX) / 2

    // Vertical: iris position within eye opening
    const le159 = landmarks[159], le145 = landmarks[145]
    const leH = Math.abs(le145.y - le159.y)
    const lRatioY = (liris.y - le159.y) / (leH + 0.001)

    const re386 = landmarks[386], re374 = landmarks[374]
    const reH = Math.abs(re374.y - re386.y)
    const rRatioY = (riris.y - re386.y) / (reH + 0.001)

    const irisY = (lRatioY + rRatioY) / 2

    // Head pose contribution (yaw/pitch affect perceived gaze direction)
    const nose = landmarks[1]
    const leftFace = landmarks[234]
    const rightFace = landmarks[454]
    const forehead = landmarks[10]
    const chin = landmarks[152]

    const faceCenter = (leftFace.x + rightFace.x) / 2
    const faceWidth = Math.abs(rightFace.x - leftFace.x)
    const headYaw = (nose.x - faceCenter) / (faceWidth + 0.001)

    const faceMidY = (forehead.y + chin.y) / 2
    const faceHeight = Math.abs(chin.y - forehead.y)
    const headPitch = (nose.y - faceMidY) / (faceHeight + 0.001)

    // Combine iris + head pose for gaze direction
    // iris offset from center (0.5) weighted more, head adds broad direction
    const rawGazeX = (irisX - 0.5) * this.scaleX + headYaw * 1.2
    const rawGazeY = (irisY - 0.5) * this.scaleY + headPitch * 1.0

    // Smooth
    const gazeX = this.gazeXEMA.update(rawGazeX)
    const gazeY = this.gazeYEMA.update(rawGazeY)

    // Map to screen coordinates (0-1 range, 0.5 = center)
    const screenX = Math.max(0, Math.min(1, 0.5 - gazeX))
    const screenY = Math.max(0, Math.min(1, 0.5 + gazeY))

    // Region label
    let region = '中央'
    if (gazeX < -0.15) region = '右'
    else if (gazeX > 0.15) region = '左'
    if (gazeY < -0.12) region = '上' + (region !== '中央' ? region : '')
    else if (gazeY > 0.12) region = '下' + (region !== '中央' ? region : '')

    const looking = Math.abs(gazeX) < 0.25 && Math.abs(gazeY) < 0.2
    return { region, looking, x: gazeX, y: gazeY, screenX, screenY }
  }
}

// ---- Focus scorer ----
class FocusScorer {
  constructor() {
    this.facingHistory = []
    this.gazeHistory = []     // track gaze stability
    this.bufferSize = 90
    this.lastGazeX = 0
    this.lastGazeY = 0
    this.gazeJitter = 0       // how much gaze jumps around
  }

  update(facing, gaze, blinkRate) {
    this.facingHistory.push(facing ? 1 : 0)
    if (this.facingHistory.length > this.bufferSize) this.facingHistory.shift()

    // Gaze stability — rapid eye movement = less focused
    if (gaze && gaze.x !== undefined) {
      const dx = Math.abs(gaze.x - this.lastGazeX)
      const dy = Math.abs(gaze.y - this.lastGazeY)
      const jitter = dx + dy
      this.gazeJitter = this.gazeJitter * 0.95 + jitter * 0.05  // EMA
      this.lastGazeX = gaze.x
      this.lastGazeY = gaze.y
    }

    const facingRatio = this.facingHistory.reduce((a, b) => a + b, 0) / this.facingHistory.length

    // Blink rate factor: 12-20/min normal. Outside = less focused
    let blinkFactor = 1
    if (blinkRate < 5) blinkFactor = 0.75   // staring, zoning out
    if (blinkRate > 28) blinkFactor = 0.65  // fatigue

    // Gaze stability factor: stable gaze = focused
    let gazeFactor = 1
    if (this.gazeJitter > 0.02) gazeFactor = 0.8   // some wandering
    if (this.gazeJitter > 0.05) gazeFactor = 0.6   // very jumpy

    const score = Math.round(facingRatio * 100 * blinkFactor * gazeFactor)

    let level = '高'
    if (score < 35) level = '低'
    else if (score < 65) level = '中'

    return { score, level }
  }
}

// ---- Synthesis ----
class SynthesisEngine {
  constructor() {
    this.lastState = ''
    this.stateStart = Date.now()
    this.events = []
  }

  synthesize(presence, attention, emotion) {
    let text = ''

    if (presence.count === 0) {
      text = '无人在屏幕前'
    } else if (presence.count > 1) {
      text = `${presence.count} 人在屏幕前`
    } else {
      const parts = []
      if (!presence.facing) {
        parts.push('没看屏幕')
      } else {
        // Don't just say "专注" — describe what they're doing
        if (attention.focus.level === '高') {
          if (attention.gaze.region !== '中央' && attention.gaze.region !== '未知') {
            parts.push(`看着屏幕${attention.gaze.region}`)
          } else {
            parts.push('注视屏幕中')
          }
        } else if (attention.focus.level === '中') {
          parts.push('在看，有点走神')
        } else {
          parts.push('心不在焉')
        }
      }

      // Only add expression if it's NOT 平静 (that's the default, not interesting)
      if (emotion.expression && !emotion.expression.includes('平静')) {
        parts.push(emotion.expression)
      }

      if (emotion.posture === '前倾') parts.push('前倾')
      else if (emotion.posture === '后仰') parts.push('后仰')
      else if (emotion.posture === '侧头') parts.push('侧头')

      text = parts.join('，') || '面对屏幕'
    }

    const stateKey = `${presence.count}-${presence.facing}-${attention.focus.level}`
    if (stateKey !== this.lastState) {
      const duration = ((Date.now() - this.stateStart) / 1000).toFixed(0)
      if (this.lastState && parseInt(duration) > 2) {
        this.events.push({
          time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
          text: `${this.lastState.startsWith('0') ? '离开' : '状态切换'} (${duration}s)`
        })
        if (this.events.length > 50) this.events.shift()
      }
      this.lastState = stateKey
      this.stateStart = Date.now()
    }

    return { text, events: this.events }
  }
}

// ---- Main engine ----
export class SenseEngine {
  constructor(video, overlayCanvas) {
    this.video = video
    this.overlay = overlayCanvas
    this.ctx = overlayCanvas.getContext('2d')

    this.faceLandmarker = null
    this.gestureRecognizer = null
    this.handLandmarker = null
    this.poseLandmarker = null
    this.imageSegmenter = null
    this.objectDetector = null
    this.faceDetector = null
    this.detector = null  // Chrome FaceDetector fallback

    this.blinkDetector = new BlinkDetector()
    this.expressionClassifier = new ExpressionClassifier()
    this.headPose = new HeadPoseEstimator()
    this.gazeEstimator = new GazeEstimator()
    this.focusScorer = new FocusScorer()
    this.synthesis = new SynthesisEngine()

    this.distanceEMA = new EMA(0.15)
    this.yawEMA = new EMA(0.2)
    this.pitchEMA = new EMA(0.2)

    this.lastResult = null
    this.frameCount = 0
    this.lastHandResult = null
    this.lastHandLmResult = null
    this.lastPoseResult = null
    this.lastSegResult = null
    this.lastObjectResult = null
    this.lastFaceDetResult = null
    this._vision = null  // cached vision module
  }

  async init() {
    // Try MediaPipe tasks-vision (self-hosted)
    try {
      await this.initMediaPipe()
      console.log('MediaPipe FaceLandmarker ready (478 landmarks + iris)')
    } catch (e) {
      console.warn('MediaPipe FaceLandmarker init failed:', e)
      // Fallback: Chrome FaceDetector
      if ('FaceDetector' in window) {
        try {
          this.detector = new window.FaceDetector({ maxDetectedFaces: 5, fastMode: true })
          console.log('Fallback: Chrome FaceDetector API')
        } catch (e2) {
          console.warn('FaceDetector failed:', e2)
          throw new Error('No face detection available')
        }
      } else {
        throw new Error('No face detection available')
      }
    }

    // Init additional recognizers in parallel (non-blocking)
    this.initGesture().catch(e => console.warn('GestureRecognizer init failed:', e))
    this.initHandLandmarker().catch(e => console.warn('HandLandmarker init failed:', e))
    this.initPose().catch(e => console.warn('PoseLandmarker init failed:', e))
    this.initSegmenter().catch(e => console.warn('ImageSegmenter init failed:', e))
    this.initObjectDetector().catch(e => console.warn('ObjectDetector init failed:', e))
    this.initFaceDetector().catch(e => console.warn('FaceDetector init failed:', e))
  }

  async initMediaPipe() {
    const vision = await import('./mediapipe/vision_bundle.mjs')
    this._vision = vision
    const { FaceLandmarker, FilesetResolver } = vision

    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')

    this.faceLandmarker = await FaceLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numFaces: 3,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false
    })
  }

  async initGesture() {
    if (!this._vision) return
    const { GestureRecognizer, FilesetResolver } = this._vision
    if (!GestureRecognizer) { console.warn('GestureRecognizer not in bundle'); return }
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')
    this.gestureRecognizer = await GestureRecognizer.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/gesture_recognizer.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 2
    })
    console.log('GestureRecognizer ready (21 landmarks × 2 hands)')
  }

  async initPose() {
    if (!this._vision) return
    const { PoseLandmarker, FilesetResolver } = this._vision
    if (!PoseLandmarker) { console.warn('PoseLandmarker not in bundle'); return }
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')
    this.poseLandmarker = await PoseLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/pose_landmarker_lite.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numPoses: 1
    })
    console.log('PoseLandmarker ready (33 body landmarks)')
  }

  async initSegmenter() {
    if (!this._vision) return
    const { ImageSegmenter, FilesetResolver } = this._vision
    if (!ImageSegmenter) { console.warn('ImageSegmenter not in bundle'); return }
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')
    this.imageSegmenter = await ImageSegmenter.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/selfie_segmenter.tflite',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false
    })
    console.log('ImageSegmenter ready (selfie segmentation)')
  }

  async initHandLandmarker() {
    if (!this._vision) return
    const { HandLandmarker, FilesetResolver } = this._vision
    if (!HandLandmarker) { console.warn('HandLandmarker not in bundle'); return }
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')
    this.handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 2
    })
    console.log('HandLandmarker ready (21 landmarks × 2 hands, no gesture)')
  }

  async initObjectDetector() {
    if (!this._vision) return
    const { ObjectDetector, FilesetResolver } = this._vision
    if (!ObjectDetector) { console.warn('ObjectDetector not in bundle'); return }
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')
    this.objectDetector = await ObjectDetector.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/efficientdet_lite0.tflite',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      maxResults: 10,
      scoreThreshold: 0.3
    })
    console.log('ObjectDetector ready (80 COCO classes)')
  }

  async initFaceDetector() {
    if (!this._vision) return
    const { FaceDetector, FilesetResolver } = this._vision
    if (!FaceDetector) { console.warn('FaceDetector not in bundle'); return }
    const wasmFileset = await FilesetResolver.forVisionTasks('./mediapipe/')
    this.faceDetector = await FaceDetector.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath: './mediapipe/blaze_face_short_range.tflite',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
    })
    console.log('FaceDetector ready (BlazeFace, fast presence check)')
  }

  detect() {
    this.frameCount++

    if (this.faceLandmarker) {
      return this.detectMediaPipe()
    } else if (this.detector) {
      return this.detectBasic()
    }
    return null
  }

  detectMediaPipe() {
    if (this.video.readyState < 2) return this.lastResult

    const now = performance.now()

    // Face landmarks (every frame)
    let faceResults
    try {
      faceResults = this.faceLandmarker.detectForVideo(this.video, now)
    } catch (e) {
      return this.lastResult
    }

    const faceCount = faceResults.faceLandmarks ? faceResults.faceLandmarks.length : 0

    // Gesture recognition (every frame if available)
    if (this.gestureRecognizer) {
      try {
        this.lastHandResult = this.gestureRecognizer.recognizeForVideo(this.video, now)
      } catch (e) { /* skip */ }
    }

    // Hand landmarks only (every frame, if gesture recognizer not available)
    if (this.handLandmarker && !this.gestureRecognizer) {
      try {
        this.lastHandLmResult = this.handLandmarker.detectForVideo(this.video, now)
      } catch (e) { /* skip */ }
    }

    // Pose landmarks (every 2nd frame to save GPU)
    if (this.poseLandmarker && this.frameCount % 2 === 0) {
      try {
        this.lastPoseResult = this.poseLandmarker.detectForVideo(this.video, now)
      } catch (e) { /* skip */ }
    }

    // Segmentation (every 3rd frame — heaviest)
    if (this.imageSegmenter && this.frameCount % 3 === 0) {
      try {
        this.lastSegResult = this.imageSegmenter.segmentForVideo(this.video, now)
      } catch (e) { /* skip */ }
    }

    // Object detection (every 5th frame — supplementary)
    if (this.objectDetector && this.frameCount % 5 === 0) {
      try {
        this.lastObjectResult = this.objectDetector.detectForVideo(this.video, now)
      } catch (e) { /* skip */ }
    }

    // Fast face detection (every frame, lightweight)
    if (this.faceDetector && this.frameCount % 4 === 0) {
      try {
        this.lastFaceDetResult = this.faceDetector.detectForVideo(this.video, now)
      } catch (e) { /* skip */ }
    }

    this.drawOverlay(faceResults, this.lastHandResult, this.lastPoseResult, this.lastObjectResult)

    if (faceCount === 0 && !this.lastHandResult?.landmarks?.length) {
      this.lastResult = this.buildResult(0, false, 0, null, null, null)
      return this.lastResult
    }

    // Primary face
    const landmarks = faceResults.faceLandmarks[0]

    // Head pose
    const pose = this.headPose.estimate(landmarks)
    pose.yaw = this.yawEMA.update(pose.yaw)
    pose.pitch = this.pitchEMA.update(pose.pitch)
    pose.facing = Math.abs(pose.yaw) < 0.15 && Math.abs(pose.pitch) < 0.2

    // Distance
    const leftEye = landmarks[33], rightEye = landmarks[263]
    const eyeDist = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y)
    const distance = this.distanceEMA.update(0.12 / (eyeDist + 0.001))

    // Gaze
    const gaze = this.gazeEstimator.estimate(landmarks)

    // Blink
    const leftEyePoints = [33, 160, 158, 133, 153, 144].map(i => landmarks[i])
    const rightEyePoints = [362, 385, 387, 263, 380, 373].map(i => landmarks[i])
    const avgEAR = (this.blinkDetector.computeEAR(leftEyePoints) + this.blinkDetector.computeEAR(rightEyePoints)) / 2
    const blinkRate = this.blinkDetector.update(avgEAR)

    // Focus
    const focus = this.focusScorer.update(pose.facing, gaze, blinkRate)

    // Expression — use blendshapes if available, otherwise landmark-based
    let expression = '😐 平静'
    if (faceResults.faceBlendshapes && faceResults.faceBlendshapes.length > 0) {
      expression = this.expressionFromBlendshapes(faceResults.faceBlendshapes[0])
    } else {
      expression = this.expressionClassifier.classify(landmarks).expression
    }

    // ---- Structured sense data (core output) ----
    const senseFrame = extractFrame(faceResults, this.lastHandResult, this.lastPoseResult, this.lastSegResult, this.lastObjectResult, this.lastFaceDetResult, this.lastHandLmResult)

    this.lastResult = this.buildResult(faceCount, pose.facing, distance,
      { gaze, blinkRate, focus },
      { expression, posture: pose.posture, tilt: pose.tilt },
      pose,
      senseFrame
    )
    return this.lastResult
  }

  expressionFromBlendshapes(blendshapes) {
    const bs = {}
    for (const cat of blendshapes.categories) {
      bs[cat.categoryName] = cat.score
    }

    const jawOpen = bs['jawOpen'] || 0
    const mouthSmileL = bs['mouthSmileLeft'] || 0
    const mouthSmileR = bs['mouthSmileRight'] || 0
    const browInnerUp = bs['browInnerUp'] || 0
    const browDownL = bs['browDownLeft'] || 0
    const browDownR = bs['browDownRight'] || 0
    const eyeSquintL = bs['eyeSquintLeft'] || 0
    const eyeSquintR = bs['eyeSquintRight'] || 0
    const mouthFrownL = bs['mouthFrownLeft'] || 0
    const mouthFrownR = bs['mouthFrownRight'] || 0
    const mouthPucker = bs['mouthPucker'] || 0
    const cheekPuff = bs['cheekPuff'] || 0
    const eyeWideL = bs['eyeWideLeft'] || 0
    const eyeWideR = bs['eyeWideRight'] || 0

    const smile = (mouthSmileL + mouthSmileR) / 2
    const frown = (mouthFrownL + mouthFrownR) / 2
    const browDown = (browDownL + browDownR) / 2
    const eyeWide = (eyeWideL + eyeWideR) / 2

    // Log top blendshapes for tuning (uncomment to debug)
    // const sorted = Object.entries(bs).filter(([,v]) => v > 0.1).sort((a,b) => b[1]-a[1]).slice(0,5)
    // if (this.frameCount % 30 === 0) console.log('BS:', sorted.map(([k,v]) => `${k}:${v.toFixed(2)}`).join(' '))

    // Thresholds tuned: jawOpen 0.4→0.6 (was triggering on slightly open mouth)
    // Need BOTH jawOpen AND eyeWide for true surprise vs just talking
    if (jawOpen > 0.6 && eyeWide > 0.3) return '😮 惊讶'
    if (jawOpen > 0.5) return '🗣️ 说话'
    if (smile > 0.5) return '😊 微笑'
    if (smile > 0.25 && jawOpen < 0.3) return '🙂 轻松'
    if (browInnerUp > 0.5 && browDown < 0.2) return '🤔 困惑'
    if (browDown > 0.4) return '🤨 皱眉'
    if (frown > 0.35) return '😕 不悦'
    if ((eyeSquintL + eyeSquintR) / 2 > 0.6) return '😑 眯眼'
    if (mouthPucker > 0.5) return '😗 嘟嘴'
    return '😐 平静'
  }

  detectBasic() {
    // Throttled FaceDetector API
    if (this.frameCount % 3 !== 0 || !this.detector) return this.lastResult
    if (this.video.readyState < 2) return this.lastResult

    this.detector.detect(this.video).then(faces => {
      const faceCount = faces.length
      if (faceCount === 0) {
        this.lastResult = this.buildResult(0, false, 0, null, null, null)
        return
      }

      const face = faces[0]
      const box = face.boundingBox
      const aspectRatio = box.width / (box.height + 1)
      const facing = aspectRatio > 0.55
      const distance = this.distanceEMA.update(200 / (box.width + 1))

      this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height)
      this.ctx.strokeStyle = 'rgba(74, 222, 128, 0.6)'
      this.ctx.lineWidth = 2
      this.ctx.strokeRect(box.x, box.y, box.width, box.height)

      this.lastResult = this.buildResult(faceCount, facing, distance, null, null, null)
    }).catch(() => {})

    return this.lastResult
  }

  drawOverlay(faceResults, handResult, poseResult, objectResult) {
    const canvas = this.overlay
    const ctx = this.ctx

    // Match canvas size to displayed video size (cover fit)
    const rect = this.video.getBoundingClientRect()
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width
      canvas.height = rect.height
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Compute cover transform
    const vw = this.video.videoWidth
    const vh = this.video.videoHeight
    const dw = rect.width
    const dh = rect.height

    const videoAspect = vw / vh
    const displayAspect = dw / dh

    let scale, offsetX, offsetY
    if (videoAspect > displayAspect) {
      scale = dh / vh; offsetX = (dw - vw * scale) / 2; offsetY = 0
    } else {
      scale = dw / vw; offsetX = 0; offsetY = (dh - vh * scale) / 2
    }

    const toX = (nx) => dw - (nx * vw * scale + offsetX)
    const toY = (ny) => ny * vh * scale + offsetY

    // ---- Face overlay ----
    if (faceResults?.faceLandmarks) {
      for (const landmarks of faceResults.faceLandmarks) {
        // Face oval
        ctx.strokeStyle = 'rgba(74, 222, 128, 0.35)'
        ctx.lineWidth = 1.5

        const ovalIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
          234, 127, 162, 21, 54, 103, 67, 109, 10]

        ctx.beginPath()
        ctx.moveTo(toX(landmarks[ovalIndices[0]].x), toY(landmarks[ovalIndices[0]].y))
        for (let i = 1; i < ovalIndices.length; i++) {
          const p = landmarks[ovalIndices[i]]
          ctx.lineTo(toX(p.x), toY(p.y))
        }
        ctx.stroke()

        // Eye corners
        ctx.fillStyle = 'rgba(96, 165, 250, 0.7)'
        for (const i of [33, 133, 362, 263]) {
          const p = landmarks[i]
          ctx.beginPath()
          ctx.arc(toX(p.x), toY(p.y), 2.5, 0, Math.PI * 2)
          ctx.fill()
        }

        // Iris
        if (landmarks.length >= 478) {
          ctx.fillStyle = 'rgba(251, 191, 36, 0.85)'
          for (const i of [468, 473]) {
            const p = landmarks[i]
            ctx.beginPath()
            ctx.arc(toX(p.x), toY(p.y), 3.5, 0, Math.PI * 2)
            ctx.fill()
          }
        }

        // Nose tip
        ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'
        const nose = landmarks[1]
        ctx.beginPath()
        ctx.arc(toX(nose.x), toY(nose.y), 2.5, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // ---- Hand overlay ----
    if (handResult?.landmarks) {
      const HAND_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],       // thumb
        [0,5],[5,6],[6,7],[7,8],       // index
        [0,9],[9,10],[10,11],[11,12],  // middle
        [0,13],[13,14],[14,15],[15,16],// ring
        [0,17],[17,18],[18,19],[19,20],// pinky
        [5,9],[9,13],[13,17]           // palm
      ]

      for (let h = 0; h < handResult.landmarks.length; h++) {
        const hand = handResult.landmarks[h]
        const color = h === 0 ? 'rgba(168, 85, 247, 0.8)' : 'rgba(236, 72, 153, 0.8)' // purple / pink

        // Connections
        ctx.strokeStyle = color.replace('0.8', '0.4')
        ctx.lineWidth = 1.5
        for (const [a, b] of HAND_CONNECTIONS) {
          ctx.beginPath()
          ctx.moveTo(toX(hand[a].x), toY(hand[a].y))
          ctx.lineTo(toX(hand[b].x), toY(hand[b].y))
          ctx.stroke()
        }

        // Joints
        ctx.fillStyle = color
        for (const pt of hand) {
          ctx.beginPath()
          ctx.arc(toX(pt.x), toY(pt.y), 2.5, 0, Math.PI * 2)
          ctx.fill()
        }

        // Gesture label
        if (handResult.gestures && handResult.gestures[h] && handResult.gestures[h].length > 0) {
          const gesture = handResult.gestures[h][0]
          if (gesture.categoryName !== 'None') {
            const wrist = hand[0]
            ctx.font = '14px "Space Mono", monospace'
            ctx.fillStyle = '#fff'
            ctx.fillText(gesture.categoryName, toX(wrist.x) + 10, toY(wrist.y) - 10)
          }
        }
      }
    }

    // ---- Pose overlay ----
    if (poseResult?.landmarks && poseResult.landmarks.length > 0) {
      const pose = poseResult.landmarks[0]
      const POSE_CONNECTIONS = [
        [11,12],                         // shoulders
        [11,13],[13,15],                 // left arm
        [12,14],[14,16],                 // right arm
        [11,23],[12,24],                 // torso
        [23,24],                         // hips
        [23,25],[25,27],                 // left leg
        [24,26],[26,28],                 // right leg
      ]

      // Connections
      ctx.strokeStyle = 'rgba(45, 212, 191, 0.4)'
      ctx.lineWidth = 2
      for (const [a, b] of POSE_CONNECTIONS) {
        if (pose[a] && pose[b]) {
          ctx.beginPath()
          ctx.moveTo(toX(pose[a].x), toY(pose[a].y))
          ctx.lineTo(toX(pose[b].x), toY(pose[b].y))
          ctx.stroke()
        }
      }

      // Joints
      ctx.fillStyle = 'rgba(45, 212, 191, 0.8)'
      for (let i = 11; i < Math.min(pose.length, 29); i++) {
        const pt = pose[i]
        if (pt) {
          ctx.beginPath()
          ctx.arc(toX(pt.x), toY(pt.y), 3, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }

    // ---- Object detection overlay ----
    if (objectResult?.detections) {
      for (const det of objectResult.detections) {
        const bb = det.boundingBox
        if (!bb) continue

        // BoundingBox is in pixel coords (not normalized), need to convert
        const x1 = toX((bb.originX + bb.width) / vw)
        const y1 = toY(bb.originY / vh)
        const x2 = toX(bb.originX / vw)
        const y2 = toY((bb.originY + bb.height) / vh)

        ctx.strokeStyle = 'rgba(251, 146, 60, 0.6)'  // orange
        ctx.lineWidth = 1.5
        ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1))

        // Label
        const cat = det.categories?.[0]
        if (cat) {
          const label = `${cat.categoryName} ${(cat.score * 100).toFixed(0)}%`
          ctx.font = '11px "Space Mono", monospace'
          ctx.fillStyle = 'rgba(251, 146, 60, 0.9)'
          ctx.fillText(label, Math.min(x1, x2), Math.min(y1, y2) - 4)
        }
      }
    }
  }

  buildResult(count, facing, distance, attention, emotion, pose, senseFrame) {
    const presence = { count, distance: distance ? distance.toFixed(1) : '-', facing }

    const attn = attention || {
      gaze: { region: '未知', looking: false },
      blinkRate: 0,
      focus: { score: 0, level: '-' }
    }

    const emo = emotion || { expression: '未知', posture: '未知', tilt: '-' }
    const synth = this.synthesis.synthesize(presence, attn, emo)

    return { presence, attention: attn, emotion: emo, synthesis: synth, pose, sense: senseFrame || null }
  }
}
