import React from 'react';
import { Heading } from '@nl-design-system-candidate/heading-react';
import { Paragraph } from '@nl-design-system-candidate/paragraph-react';
import { Button } from '@nl-design-system-candidate/button-react';
import { Link } from '@nl-design-system-candidate/link-react';
import { Code } from '@nl-design-system-candidate/code-react';
import type { CitationResponse } from '../types/gpo';
import { getVerifiedPdfUrl } from '../api/client';

interface CitationDrawerProps {
  citation: CitationResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({
  citation,
  loading,
  error,
  onClose,
}) => {
  if (!citation && !loading && !error) return null;

  return (
    <div className="vr-drawer-backdrop" onClick={onClose} role="presentation">
      <aside
        className="vr-drawer"
        role="dialog"
        aria-labelledby="vr-drawer-title"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="vr-drawer__header">
          <Heading level={2} id="vr-drawer-title" className="vr-drawer__title">
            Geverifieerde bronpassage
          </Heading>
          <Button
            purpose="secondary"
            onClick={onClose}
            aria-label="Sluit paneel"
          >
            Sluiten ✕
          </Button>
        </div>

        {loading && (
          <Paragraph>Bezig met ophalen en controleren van het officiële broncitaat…</Paragraph>
        )}

        {error && (
          <div style={{ color: '#b91c1c', padding: '1rem', backgroundColor: '#fef2f2', borderRadius: '4px' }}>
            <strong>Fout:</strong> {error}
          </div>
        )}

        {citation && (
          <>
            <div className="vr-drawer__quote">
              <span style={{ display: 'block', fontSize: '13px', fontWeight: 700, color: '#383284', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
                Exacte bronzin (geverifieerd op karakter-offset):
              </span>
              <blockquote>“{citation.quote}”</blockquote>
            </div>

            <div className="vr-drawer__context">
              <Heading level={3} className="vr-drawer__context-title">
                Context in uitspraak ({citation.heading})
              </Heading>
              <Paragraph>{citation.text}</Paragraph>
            </div>

            <div className="vr-drawer__meta">
              <div><strong>Zaaknummer:</strong> {citation.case_number}</div>
              <div><strong>Documentpagina:</strong> Pagina {citation.page_number}</div>
              <div><strong>Adviesdatum:</strong> {citation.advice_date || 'Niet vermeld'}</div>
              <div>
                <strong>PDF SHA-256:</strong> <Code>{citation.pdf_sha256.slice(0, 16)}…</Code>
              </div>
              <div style={{ marginTop: '0.5rem' }}>
                <Link
                  href={citation.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="vr-external-link"
                >
                  Open officiële PDF op Onderwijsgeschillen.nl (p. {citation.page_number}) ↗
                </Link>
              </div>
              <div>
                <Link
                  href={`${getVerifiedPdfUrl(citation.version_id)}#page=${citation.page_number}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="vr-external-link"
                >
                  Lokale geverifieerde PDF-snapshot raadplegen ↗
                </Link>
              </div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
};
