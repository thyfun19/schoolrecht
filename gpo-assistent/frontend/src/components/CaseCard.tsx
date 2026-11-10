import React, { useState } from 'react';
import { Heading } from '@nl-design-system-candidate/heading-react';
import { Paragraph } from '@nl-design-system-candidate/paragraph-react';
import { Button } from '@nl-design-system-candidate/button-react';
import { Link } from '@nl-design-system-candidate/link-react';
import { DataBadge } from '@nl-design-system-candidate/data-badge-react';
import type { PineconeCaseResult } from '../types/gpo';

interface CaseCardProps {
  item: PineconeCaseResult;
  searchQuery?: string;
  rank?: number;
  onSelectCitation: (processingId: string, sourceId: string) => void;
}

export const CaseCard: React.FC<CaseCardProps> = ({ item, searchQuery = '', rank, onSelectCitation }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Format date in natural Dutch: "15 mei 2023"
  const rawDate = item.advice_date || item.publication_date || '';
  let formattedDate = rawDate;
  if (rawDate && rawDate.includes('-')) {
    const parts = rawDate.split('-');
    if (parts.length === 3) {
      const year = parts[0];
      const monthNum = parseInt(parts[1], 10);
      const day = parseInt(parts[2], 10);
      const months = [
        'januari', 'februari', 'maart', 'april', 'mei', 'juni',
        'juli', 'augustus', 'september', 'oktober', 'november', 'december'
      ];
      const monthName = months[monthNum - 1] || parts[1];
      formattedDate = `${day} ${monthName} ${year}`;
    }
  }

  // Clean title
  const rawTitle = (item.title || `Zaak ${item.case_number}: ${item.topic || 'Geschil passend onderwijs'}`)
    .replace(/reel/g, 'reëel')
    .replace(/gevalueerd/g, 'geëvalueerd')
    .replace(/AcAcn/g, 'één');

  const narrativeSummary = item.preview?.quote || item.matched_chunks[0]?.text || item.summary || item.conclusies?.[0]?.text || '';
  const citationId = item.preview?.source_id || item.matched_chunks[0]?.source_ids?.[0];
  const pageNumber = item.preview?.page_number || item.matched_chunks[0]?.page;

  // Highlight matched search keywords
  const highlightWords = (text: string) => {
    if (!searchQuery.trim()) return text;
    const keywords = searchQuery
      .split(/\s+/)
      .map((w) => w.trim().replace(/[.,?!;:()"']/g, '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .filter((w) => w.length > 3);

    if (keywords.length === 0) return text;
    const regex = new RegExp(`(${keywords.join('|')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, i) =>
      keywords.some((kw) => kw.toLowerCase() === part.toLowerCase()) ? (
        <mark key={i} className="vr-text-highlight">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <article className="vr-case-card" aria-labelledby={`case-title-${item.case_number}`}>
      <div className="vr-case-card__top-row">
        <div className="vr-case-card__badges">
          {rank !== undefined && <span className="vr-case-card__rank">#{rank}</span>}
          <DataBadge>Zaak {item.case_number}</DataBadge>
          {item.sector && <DataBadge>{item.sector}</DataBadge>}
          {pageNumber && <DataBadge>Pagina {pageNumber}</DataBadge>}
        </div>
        {formattedDate && <div className="vr-case-card__date">Uitspraak: {formattedDate}</div>}
      </div>

      <Heading level={3} id={`case-title-${item.case_number}`} className="vr-case-card__title">
        {highlightWords(rawTitle)}
      </Heading>

      <div className="vr-case-card__summary">
        <div className="vr-case-card__quote-label">
          Geselecteerde overweging van de Commissie{pageNumber ? ` (p. ${pageNumber})` : ''}:
        </div>
        {isExpanded ? (
          <Paragraph className="vr-case-card__quote-text">“{highlightWords(narrativeSummary)}”</Paragraph>
        ) : (
          <Paragraph className="vr-case-card__quote-text">
            “{highlightWords(
              item.preview?.excerpt ?? (narrativeSummary.length > 280
                ? `${narrativeSummary.slice(0, 275)}…`
                : narrativeSummary)
            )}”
          </Paragraph>
        )}
      </div>

      {/* Uitgeklapte staat: oordeel en overwegingen */}
      {isExpanded && (
        <div className="vr-case-card__full-content">
          {item.conclusies && item.conclusies.length > 0 && (
            <div style={{ marginBottom: '1rem' }}>
              <Heading level={4}>Oordeel van de Commissie:</Heading>
              <Paragraph>{item.conclusies.map((c) => c.text).join(' ')}</Paragraph>
            </div>
          )}

          {item.commissieoverwegingen && item.commissieoverwegingen.length > 0 && (
            <div>
              <Heading level={4}>Omliggende overwegingen uit de uitspraak:</Heading>
              {item.commissieoverwegingen.slice(0, 3).map((sec) => (
                <Paragraph key={sec.section_key} style={{ marginBottom: '0.6rem' }}>
                  <strong>Pagina {sec.page_number}:</strong> {sec.text}
                </Paragraph>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Actiebalk onderkant */}
      <div className="vr-case-card__actions">
        <div className="vr-case-card__actions-left">
          <Button
            purpose="subtle"
            className="vr-link-button"
            onClick={() => setIsExpanded((prev) => !prev)}
            aria-expanded={isExpanded}
          >
            {isExpanded ? '▲ Toon minder context' : '▼ Toon meer context uit deze zaak'}
          </Button>

          {citationId && (
            <Button
              purpose="subtle"
              className="vr-link-button vr-link-button--source"
              onClick={() => onSelectCitation(item.processing_id, citationId)}
              aria-label={`Controleer bronpassage voor zaak ${item.case_number}`}
            >
              Controleer broncitaat{pageNumber ? ` (p. ${pageNumber})` : ''}
            </Button>
          )}
        </div>

        {item.source_url ? (
          <Link
            href={item.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="vr-external-link"
          >
            Officiële PDF openen ↗
          </Link>
        ) : item.detail_url ? (
          <Link
            href={item.detail_url}
            target="_blank"
            rel="noopener noreferrer"
            className="vr-external-link"
          >
            Officiële uitspraak openen ↗
          </Link>
        ) : null}
      </div>
    </article>
  );
};
