import type { PineconeCaseResult, PineconeSearchResponse } from '../types/gpo';
import type { ChatCitationRef } from '../types/chat';

export interface SynthesizedAnswer {
  summary: string;
  points: { text: string; citationRef?: ChatCitationRef }[];
  citations: ChatCitationRef[];
  closing: string;
}

export function synthesizeAnswer(
  query: string,
  searchResponse: PineconeSearchResponse
): SynthesizedAnswer {
  const cases: PineconeCaseResult[] = searchResponse.results || [];
  const citations: ChatCitationRef[] = [];

  if (cases.length === 0) {
    return {
      summary: 'Ik heb in het GPO-bronregister gezocht, maar geen specifieke uitspraken gevonden die direct aansluiten bij deze formulering.',
      points: [
        {
          text: 'Probeer uw vraag te herformuleren met juridische kernbegrippen zoals "ontwikkelingsperspectief (OPP)", "toelaatbaarheidsverklaring (TLV)", "zorgplicht", "handelingsverlegenheid" of "verwijdering".',
        },
      ],
      citations: [],
      closing: 'U kunt ook filteren op een specifiek onderwerp of onderwijssector.',
    };
  }

  // Collect key findings from conclusions and matched chunks
  const points: { text: string; citationRef?: ChatCitationRef }[] = [];

  cases.slice(0, 3).forEach((item, index) => {
    const caseNum = item.case_number;
    const procId = item.processing_id;

    // First preference: conclusions
    if (item.conclusies && item.conclusies.length > 0) {
      const conc = item.conclusies[0];
      const chunkWithSource = item.matched_chunks.find(
        (c) => c.source_ids && c.source_ids.length > 0
      );
      const sourceId = chunkWithSource?.source_ids?.[0] || `src-${procId}-1`;

      const citRef: ChatCitationRef = {
        sourceId,
        processingId: procId,
        caseNumber: caseNum,
        pageNumber: conc.page_number,
        label: `Zaak ${caseNum} (p. ${conc.page_number})`,
      };
      citations.push(citRef);

      const snippet = conc.text.length > 240 ? `${conc.text.slice(0, 237)}…` : conc.text;
      points.push({
        text: `Zaak ${caseNum} (${item.topic || 'Passend onderwijs'}, ${item.advice_date || item.publication_date || 'datum onbekend'}): ${snippet}`,
        citationRef: citRef,
      });
    } else if (item.matched_chunks.length > 0) {
      const topChunk = item.matched_chunks[0];
      const sourceId = topChunk.source_ids?.[0] || `src-${procId}-1`;
      const citRef: ChatCitationRef = {
        sourceId,
        processingId: procId,
        caseNumber: caseNum,
        pageNumber: topChunk.page,
        label: `Zaak ${caseNum} (p. ${topChunk.page ?? '?'})`,
      };
      citations.push(citRef);

      const text = topChunk.text || 'Relevante passage uit de commissieoverwegingen.';
      const snippet = text.length > 240 ? `${text.slice(0, 237)}…` : text;
      points.push({
        text: `Zaak ${caseNum}: "${snippet}"`,
        citationRef: citRef,
      });
    }
  });

  const summary = `Op basis van ${cases.length} relevante uitspraken van de Geschillencommissie Passend Onderwijs (GPO) over "${query}":`;
  const closing =
    'Elk standpunt is direct herleidbaar naar de originele commissieoverwegingen. Klik op een bronverwijzing om de exacte gecontroleerde zin en PDF-pagina te bekijken.';

  return { summary, points, citations, closing };
}
