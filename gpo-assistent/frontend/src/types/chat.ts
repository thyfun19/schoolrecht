import type { CitationResponse, PineconeCaseResult } from './gpo';

export interface ChatCitationRef {
  sourceId: string;
  processingId: string;
  caseNumber: string;
  pageNumber?: number;
  label: string;
}

export interface ChatMessageItem {
  id: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  text: string;
  needsClarification?: boolean;
  clarificationPrompt?: string;
  suggestedReplies?: string[];
  results?: PineconeCaseResult[];
  citations?: ChatCitationRef[];
  method?: string;
}
