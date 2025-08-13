'use client'

import { useState } from 'react'
import { ChatBubble } from './ChatBubble'
import { SourceHint } from './SourceHint'
import { PIPELINE_VERSION } from '../lib/constants'

const MAX_MESSAGES = 20
const NEED_CITATION = true

export function AskAIDialog({
  open,
  apiBase,
  onMinimize,
  onEnd
}: {
  open: boolean
  apiBase: string
  onMinimize: () => void
  onEnd: () => void
}) {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<{ sender: 'user' | 'ai'; text: string }[]>([])
  const [sources, setSources] = useState<any[]>([])
  const [summary, setSummary] = useState('')

  function stripHtml(html: string) {
    return html.replace(/<[^>]+>/g, '')
  }

  function summarize(existing: string, msgs: { text: string }[]) {
    if (msgs.length === 0) return existing
    const addition = msgs.map(m => stripHtml(m.text)).join(' ')
    const combined = existing ? `${existing} ${addition}` : addition
    return combined.slice(-1000)
  }

  function renderMarkdown(text: string) {
    // code blocks
    let html = text.replace(
      /```([\s\S]*?)```/g,
      (_, code) =>
        `<pre class="bg-gray-100 p-2 rounded overflow-x-auto"><code>${code
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')}</code></pre>`
    )

    // inline code
    html = html.replace(
      /`([^`]+)`/g,
      (_, code) =>
        `<code class="bg-gray-100 rounded px-1">${code
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')}</code>`
    )

    // headings
    html = html
      .replace(/^###### (.*)$/gm, '<h6 class="font-semibold">$1</h6>')
      .replace(/^##### (.*)$/gm, '<h5 class="font-semibold">$1</h5>')
      .replace(/^#### (.*)$/gm, '<h4 class="font-semibold">$1</h4>')
      .replace(/^### (.*)$/gm, '<h3 class="font-semibold">$1</h3>')
      .replace(/^## (.*)$/gm, '<h2 class="font-semibold">$1</h2>')
      .replace(/^# (.*)$/gm, '<h1 class="font-semibold">$1</h1>')

    // bold & italics
    html = html
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')

    // links
    html = html.replace(
      /\[(.+?)\]\((.+?)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 underline">$1</a>'
    )

    // blockquotes
    html = html.replace(
      /^> (.*)$/gm,
      '<blockquote class="border-l-4 pl-4 italic text-gray-600">$1</blockquote>'
    )

    // unordered lists
    html = html.replace(/^(?:[-+*] .*(?:\n|$))+?/gm, match => {
      const items = match
        .trim()
        .split('\n')
        .map(line => line.replace(/^[-+*] /, '').trim())
      return `<ul class="list-disc pl-5 space-y-1">${items
        .map(item => `<li>${item}</li>`)
        .join('')}</ul>`
    })

    // ordered lists
    html = html.replace(/^(?:\d+\. .*(?:\n|$))+?/gm, match => {
      const items = match
        .trim()
        .split('\n')
        .map(line => line.replace(/^\d+\. /, '').trim())
      return `<ol class="list-decimal pl-5 space-y-1">${items
        .map(item => `<li>${item}</li>`)
        .join('')}</ol>`
    })

    // line breaks
    return html.replace(/\n+/g, '<br />')
  }

  async function handleAsk() {
    if (!question) return

    const userMessage = { sender: 'user' as const, text: renderMarkdown(question) }
    const overflow = messages.slice(0, Math.max(0, messages.length - (MAX_MESSAGES - 1)))
    const newSummary = summarize(summary, overflow)
    const history = [...messages.slice(-MAX_MESSAGES + 1), userMessage]
    setSummary(newSummary)
    setMessages(history)
    setQuestion('')

    try {
      const body = JSON.stringify({
        question,
        history,
        summary: newSummary,
        user_confidence: 1,
        history_len: history.length,
        need_citation: NEED_CITATION
      })
      const res = await fetch(`${apiBase}/api/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Pipeline-Version': PIPELINE_VERSION },
        body
      })

      if (!res.ok) throw new Error('Request failed')

      const data = await res.json()
      const retrieved = data.chunks || []
      setSources(retrieved)

      let answer = data.answer as string

      if (!answer || retrieved.length === 0) {
        try {
          const aiRes = await fetch(`${apiBase}/api/askai`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Pipeline-Version': PIPELINE_VERSION },
            body: JSON.stringify({ question, history, summary: newSummary })
          })

          if (aiRes.ok) {
            const aiData = await aiRes.json()
            answer = aiData.answer as string
          }
        } catch (err) {
          // ignore and use generic message below
        }

        if (!answer) {
          answer = 'Sorry, I could not find an answer at this time.'
        }
      }

      const aiMessage = { sender: 'ai' as const, text: renderMarkdown(answer) }
      setMessages(prev => {
        const newMessages = [...prev, aiMessage]
        const drop = newMessages.slice(0, Math.max(0, newMessages.length - MAX_MESSAGES))
        if (drop.length) {
          setSummary(s => summarize(s, drop))
        }
        return newMessages.slice(-MAX_MESSAGES)
      })
    } catch (err) {
      const aiMessage = { sender: 'ai' as const, text: renderMarkdown('Something went wrong. Please try again later.') }
      setMessages(prev => {
        const newMessages = [...prev, aiMessage]
        const drop = newMessages.slice(0, Math.max(0, newMessages.length - MAX_MESSAGES))
        if (drop.length) {
          setSummary(s => summarize(s, drop))
        }
        return newMessages.slice(-MAX_MESSAGES)
      })
    }
  }

  function handleEnd() {
    setMessages([])
    setQuestion('')
    setSources([])
    onEnd()
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="hidden md:block flex-1 bg-black/40" onClick={onMinimize} />
      <div className="w-full md:w-1/2 h-full bg-white shadow-xl flex flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Ask anything about your docs</h2>
          <div className="flex gap-2 text-gray-500">
            <button onClick={onMinimize} className="hover:text-gray-800" title="Minimize">
              –
            </button>
            <button onClick={handleEnd} className="hover:text-gray-800" title="End conversation">
              End
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 text-gray-800 space-y-4">
          {messages.map((m, idx) => (
            <ChatBubble key={idx} message={m.text} type={m.sender} />
          ))}
          {sources.length > 0 && <SourceHint sources={sources} />}
        </div>

        <div className="border-t p-4">
          <textarea
            className="w-full border p-3 rounded-lg mb-4 text-black"
            rows={3}
            placeholder="Type your question..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleAsk()
              }
            }}
          />
          <button
            onClick={handleAsk}
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700"
          >
            Ask
          </button>
        </div>
      </div>
    </div>
  )
}
