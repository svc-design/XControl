import { PIPELINE_VERSION } from './constants'

export async function fetchRelatedDocs(query: string): Promise<string[]> {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080'
  try {
    const res = await fetch(`${apiBase}/api/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Pipeline-Version': PIPELINE_VERSION },
      body: JSON.stringify({
        question: query,
        user_confidence: 1,
        history_len: 0,
        need_citation: false
      })
    })
    if (!res.ok) return []
    const data = await res.json()
    return (data.chunks || []).map((c: any) => `${c.repo}:${c.path}`)
  } catch (err) {
    console.warn('fetchRelatedDocs failed', err)
    return []
  }
}
