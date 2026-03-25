import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

let transcriber = null;

self.onmessage = async (e) => {
  const { type, audio } = e.data;

  if (type === 'init') {
    self.postMessage({ type: 'status', status: 'loading', message: '加载语音模型...' });
    try {
      transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base', {
        device: 'webgpu',
        dtype: 'fp32',
      });
      self.postMessage({ type: 'status', status: 'ready', message: '模型就绪' });
    } catch (err) {
      self.postMessage({ type: 'status', status: 'error', message: `模型加载失败: ${err.message}` });
    }
  }

  if (type === 'transcribe' && transcriber) {
    try {
      const result = await transcriber(audio, {
        language: 'chinese',
        task: 'transcribe',
        chunk_length_s: 5,
      });
      self.postMessage({ type: 'result', text: result.text, chunks: result.chunks });
    } catch (err) {
      console.error('Whisper transcribe error:', err);
    }
  }
};
