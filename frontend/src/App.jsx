// frontend/src/App.jsx
import { useEffect, useRef, useState } from 'react'
import axios from 'axios'

export default function App() {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState(() => JSON.parse(localStorage.getItem('history') || '[]'))

  useEffect(() => {
    const c = canvasRef.current
    const ctx = c.getContext('2d')
    // 黑底白字，和 MNIST 一致
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, c.width, c.height)
    ctx.lineWidth = 18
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.strokeStyle = '#fff'
  }, [])

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top
    return { x, y }
  }

  const start = (e) => {
    setIsDrawing(true)
    const { x, y } = getPos(e)
    const ctx = canvasRef.current.getContext('2d')
    ctx.beginPath()
    ctx.moveTo(x, y)
  }
  const draw = (e) => {
    if (!isDrawing) return
    const { x, y } = getPos(e)
    const ctx = canvasRef.current.getContext('2d')
    ctx.lineTo(x, y)
    ctx.stroke()
  }
  const end = () => setIsDrawing(false)

  const clearCanvas = () => {
    const c = canvasRef.current
    const ctx = c.getContext('2d')
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, c.width, c.height)
    setResult(null)
  }

  const downscaleTo28 = () => {
    const big = canvasRef.current
    const off = document.createElement('canvas')
    off.width = 28
    off.height = 28
    const offctx = off.getContext('2d')
    offctx.drawImage(big, 0, 0, 28, 28)
    const img = offctx.getImageData(0, 0, 28, 28).data
    const pixels = []
    for (let i = 0; i < img.length; i += 4) {
      const r = img[i], g = img[i + 1], b = img[i + 2]
      const gray = (r + g + b) / 3 // 0..255
      pixels.push(gray)
    }
    return pixels
  }

  const recognize = async () => {
    setResult({ loading: true })
    const pixels = downscaleTo28()
    try {
      const { data } = await axios.post('http://127.0.0.1:5000/upload', { pixels })
      if (!data.ok) throw new Error(data.error || 'Server error')
      const top = topK(data.probs, 3)
      const r = { pred: data.pred, top, time_ms: data.time_ms }
      setResult(r)
      const rec = { ...r, at: new Date().toLocaleString() }
      const newHist = [rec, ...history].slice(0, 10)
      setHistory(newHist)
      localStorage.setItem('history', JSON.stringify(newHist))
    } catch (e) {
      setResult({ error: e.message })
    }
  }

  const topK = (arr, k) => {
    const pairs = arr.map((p, i) => [i, p])
    pairs.sort((a, b) => b[1] - a[1])
    return pairs.slice(0, k).map(([i, p]) => ({ digit: i, prob: p }))
  }

  const onUploadImage = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const img = new Image()
    img.onload = () => {
      const c = canvasRef.current
      const ctx = c.getContext('2d')
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, c.width, c.height)
      const scale = Math.min(c.width / img.width, c.height / img.height)
      const w = img.width * scale
      const h = img.height * scale
      const x = (c.width - w) / 2
      const y = (c.height - h) / 2
      ctx.drawImage(img, x, y, w, h)
    }
    img.src = URL.createObjectURL(file)
    e.target.value = null
  }

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto', maxWidth: 820, margin: '24px auto', padding: 16 }}>
      <h1>手写数字识别</h1>
      <p>在画布上写 0~9（尽量居中，长度占据画布70~90%），或上传图片（黑底白字，粗体数字）。</p>

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          style={{ border: '1px solid #ddd', borderRadius: 8, touchAction: 'none' }}
          onMouseDown={start}
          onMouseMove={draw}
          onMouseUp={end}
          onMouseLeave={end}
          onTouchStart={(e) => { e.preventDefault(); start(e) }}
          onTouchMove={(e) => { e.preventDefault(); draw(e) }}
          onTouchEnd={(e) => { e.preventDefault(); end() }}
        />
        <div style={{ minWidth: 280 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <button onClick={clearCanvas}>清空</button>
            <button onClick={recognize}>识别</button>
            <label>
              <input type="file" accept="image/*" onChange={onUploadImage} style={{ display: 'none' }} />
              <span style={{ padding: '6px 12px', border: '1px solid #ccc', borderRadius: 4, cursor: 'pointer' }}>上传图片</span>
            </label>
          </div>
          {result?.loading && <p>识别中…</p>}
          {result?.error && <p style={{ color: 'crimson' }}>错误：{result.error}</p>}
          {result?.pred !== undefined && (
            <div>
              <h3>预测：{result.pred} <small style={{ color: '#666' }}>({result.time_ms} ms)</small></h3>
              <ol>
                {result.top.map(t => (
                  <li key={t.digit}>{t.digit}: {(t.prob * 100).toFixed(1)}%</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        <h3>历史（保存在本地浏览器）</h3>
        <ul>
          {history.map((h, idx) => (
            <li key={idx}>{h.at} → {h.pred}（Top1: {h.top[0].digit} {(h.top[0].prob*100).toFixed(0)}%）</li>
          ))}
        </ul>
      </div>
    </div>
  )
}
