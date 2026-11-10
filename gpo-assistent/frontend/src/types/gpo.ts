export interface RegistryCounts {
  documents: number;
  pdf_files: number;
  document_versions: number;
  extractions: number;
  pages: number;
  sections: number;
  citations: number;
  availability: Record<string, number>;
}

export interface HealthResponse {
  status: 'ok';
  schema_revision: string;
  counts: RegistryCounts;
  pinecone_enabled: boolean;
  answer_generation: boolean;
  paid_api_enabled: boolean;
}

export interface MatchedChunk {
  chunk_id: string;
  score: number;
  page?: number;
  section?: string;
  role?: string;
  text?: string;
  source_ids?: string[];
}

export interface SectionRecord {
  processing_id: string;
  section_key: string;
  page_number: number;
  heading: string;
  role: string;
  category: string;
  text: string;
}

export interface PineconeCaseResult {
  preview?: { excerpt?: string; quote: string; source_id: string; page_number: number };
  case_number: string;
  processing_id: string;
  version_id?: string;
  score: number;
  title?: string | null;
  summary?: string | null;
  topic?: string | null;
  sector?: string | null;
  publication_date?: string | null;
  advice_date?: string | null;
  detail_url?: string | null;
  source_url?: string | null;
  matched_chunks: MatchedChunk[];
  conclusies?: SectionRecord[];
  commissieoverwegingen?: SectionRecord[];
}

export interface PineconeSearchResponse {
  status?: 'needs_clarification' | 'needs_additional_sources' | 'insufficient_sources';
  clarification_question?: string;
  reason?: string;
  method?: string;
  index_name?: string;
  cases?: string[];
  processing_ids?: string[];
  results?: PineconeCaseResult[];
  scope?: string;
}

export interface CitationResponse {
  processing_id: string;
  source_id: string;
  section_key: string;
  start_offset: number;
  end_offset: number;
  quote: string;
  page_number: number;
  heading: string;
  role: string;
  text: string;
  version_id: string;
  advice_date?: string | null;
  pdf_sha256: string;
  source_url: string;
  case_number: string;
  pdf_url: string;
  semantic_support: string;
}
