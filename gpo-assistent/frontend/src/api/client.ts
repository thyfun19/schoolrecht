import type { CitationResponse, HealthResponse, PineconeSearchResponse } from '../types/gpo';

const API_BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? '' : 'https://schoolrecht-api.onrender.com');

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/v1/health`);
  if (!res.ok) {
    throw new Error('Kan de status van het GPO-bronregister momenteel niet ophalen.');
  }
  return res.json();
}

export async function searchPinecone(params: {
  question: string;
  limit?: number;
  topic?: string;
  sector?: string;
}): Promise<PineconeSearchResponse> {
  const res = await fetch(`${API_BASE}/api/v1/search/pinecone`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: params.question,
      limit: params.limit ?? 5,
      topic: params.topic || null,
      sector: params.sector || null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Zoekopdracht kon niet worden uitgevoerd.');
  }
  return res.json();
}

export async function fetchCitation(processingId: string, sourceId: string): Promise<CitationResponse> {
  const url = new URL(`${API_BASE}/api/v1/sources/${encodeURIComponent(sourceId)}`, window.location.origin);
  url.searchParams.set('processing_id', processingId);
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error('Broncitaat kon niet worden opgehaald of geverifieerd.');
  }
  return res.json();
}

export function getVerifiedPdfUrl(versionId: string): string {
  return `${API_BASE}/api/v1/documents/${encodeURIComponent(versionId)}/pdf`;
}
