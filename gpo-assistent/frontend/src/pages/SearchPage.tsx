import React, { useEffect, useState } from 'react';
import { Heading } from '@nl-design-system-candidate/heading-react';
import { Paragraph } from '@nl-design-system-candidate/paragraph-react';
import { Button } from '@nl-design-system-candidate/button-react';
import { Link } from '@nl-design-system-candidate/link-react';
import { DataBadge } from '@nl-design-system-candidate/data-badge-react';
import { SkipLink } from '@nl-design-system-candidate/skip-link-react';
import { Code } from '@nl-design-system-candidate/code-react';
import {
  fetchHealth,
  searchPinecone,
  fetchCitation,
} from '../api/client';
import type {
  CitationResponse,
  HealthResponse,
  PineconeSearchResponse,
} from '../types/gpo';
import { CaseCard } from '../components/CaseCard';
import { CitationDrawer } from '../components/CitationDrawer';

const EXAMPLE_QUESTIONS = [
  {
    label: 'TLV & onafhankelijke deskundigen',
    question:
      'Welke eisen stelt de commissie aan de twee onafhankelijke deskundigenadviezen bij een toelaatbaarheidsverklaring (TLV)?',
    topic: '',
  },
  {
    label: 'Ontwikkelingsperspectief (OPP) & OOGO',
    question:
      'Wat houdt op overeenstemming gericht overleg (OOGO) met ouders in bij het opstellen en evalueren van een ontwikkelingsperspectief (OPP)?',
    topic: '',
  },
  {
    label: 'Verwijdering & zorgplicht andere school',
    question:
      'Wanneer mag het bevoegd gezag een leerling met extra ondersteuningsbehoefte definitief verwijderen onder de zorgplicht passend onderwijs?',
    topic: '',
  },
  {
    label: 'Weigering toelating & ondersteuningsprofiel',
    question:
      'Wanneer mag een school weigeren een leerling met een extra ondersteuningsbehoefte toe te laten vanwege handelingsverlegenheid?',
    topic: '',
  },
];

export const SearchPage: React.FC = () => {
  const [question, setQuestion] = useState<string>('');
  const [selectedTopic, setSelectedTopic] = useState<string>('');
  const [selectedSector, setSelectedSector] = useState<string>('');
  const [limit, setLimit] = useState<number>(5);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [searchResult, setSearchResult] = useState<PineconeSearchResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [activeCitation, setActiveCitation] = useState<CitationResponse | null>(null);
  const [citationLoading, setCitationLoading] = useState<boolean>(false);
  const [citationError, setCitationError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const executeSearch = async (
    queryText: string,
    topicOverride?: string,
    sectorOverride?: string
  ) => {
    const trimmed = queryText.trim();
    if (trimmed.length < 3) {
      setErrorMessage(
        'Typ een vraag of zoekterm van minimaal 3 tekens (bijvoorbeeld: "ontwikkelingsperspectief evalueren met ouders").'
      );
      return;
    }

    setLoading(true);
    setErrorMessage(null);

    const activeTopic = topicOverride !== undefined ? topicOverride : selectedTopic;
    const activeSector = sectorOverride !== undefined ? sectorOverride : selectedSector;

    try {
      const response = await searchPinecone({
        question: trimmed,
        limit,
        topic: activeTopic || undefined,
        sector: activeSector || undefined,
      });
      setSearchResult(response);
    } catch (err) {
      setErrorMessage(
        err instanceof Error
          ? err.message
          : 'Er ging iets mis bij het doorzoeken van de GPO-uitspraken. Controleer de verbinding en probeer het opnieuw.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void executeSearch(question);
  };

  const handleExampleClick = (exampleQuestion: string, topic: string) => {
    setQuestion(exampleQuestion);
    setSelectedTopic(topic);
    void executeSearch(exampleQuestion, topic, selectedSector);
  };

  const handleSelectCitation = async (processingId: string, sourceId: string) => {
    setCitationLoading(true);
    setCitationError(null);
    setActiveCitation(null);
    try {
      const citation = await fetchCitation(processingId, sourceId);
      setActiveCitation(citation);
    } catch (err) {
      setCitationError(
        err instanceof Error
          ? err.message
          : 'Het broncitaat kon niet worden opgehaald of geverifieerd.'
      );
    } finally {
      setCitationLoading(false);
    }
  };

  return (
    <div className="nlds-page">
      <SkipLink href="#main-content">Direct naar hoofdinhoud</SkipLink>

      {/* Rijksoverheid / NL Design System Top Header */}
      <header className="nlds-header" role="banner">
        <div className="nlds-container nlds-header__inner">
          <div className="nlds-header__brand">
            <span className="nlds-header__Crest" aria-hidden="true">
              ⚖️
            </span>
            <div>
              <Heading level={1} className="nlds-header__title">
                GPO-Assistent — Geschillencommissie Passend Onderwijs
              </Heading>
              <Paragraph className="nlds-header__subtitle">
                Verifieerbare jurisprudentie-verkenner met broncitaten (&ldquo;Data bij de bron&rdquo;
                &bull; NL Design System &bull; WCAG 2.1 AA)
              </Paragraph>
            </div>
          </div>

          <div className="nlds-header__status" aria-live="polite">
            {health ? (
              <div className="nlds-status-badges">
                <DataBadge className="nlds-badge nlds-badge--online">
                  Supabase: {health.counts.documents} uitspraken ({health.counts.citations} citaten)
                </DataBadge>
                <DataBadge
                  className={
                    health.pinecone_enabled
                      ? 'nlds-badge nlds-badge--pinecone'
                      : 'nlds-badge nlds-badge--neutral'
                  }
                >
                  {health.pinecone_enabled
                    ? 'Pinecone BGE-M3 (1024d): Actief'
                    : 'Lexicale modus'}
                </DataBadge>
              </div>
            ) : (
              <DataBadge className="nlds-badge nlds-badge--neutral">
                Verbinding met GPO-bronregister controleren…
              </DataBadge>
            )}
          </div>
        </div>
      </header>

      <main id="main-content" className="nlds-container nlds-main" role="main">
        {/* Search Panel */}
        <section className="nlds-search-panel" aria-labelledby="search-heading">
          <Heading level={2} id="search-heading">
            Zoek in uitspraken van de Geschillencommissie Passend Onderwijs (GPO)
          </Heading>
          <Paragraph>
            Stel een juridische of casusgerichte vraag in begrijpelijke taal (B1). Elke gevonden
            uitspraak toont direct de relevante passages uit de commissieoverwegingen, het formele
            oordeel en SHA-256-geverifieerde bronzinnen uit de originele PDF van
            Onderwijsgeschillen.nl.
          </Paragraph>

          <form onSubmit={handleFormSubmit} className="nlds-search-form" noValidate>
            <div className="nlds-form-field">
              <label htmlFor="gpo-question-input" className="nlds-label">
                Uw onderzoeksvraag of casusbeschrijving
              </label>
              <textarea
                id="gpo-question-input"
                className="nlds-textarea"
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Bijv.: Mag een samenwerkingsverband een TLV afgeven als één van de twee deskundigen niet onafhankelijk is?"
                aria-describedby="gpo-question-hint"
                required
              />
              <span id="gpo-question-hint" className="nlds-hint">
                Geef bij voorkeur het onderwerp (zoals TLV, OPP, toelating of verwijdering) en de
                situatie aan.
              </span>
            </div>

            <div className="nlds-filters-grid">
              <div className="nlds-form-field">
                <label htmlFor="gpo-topic-select" className="nlds-label">
                  Onderwerp (optioneel filter)
                </label>
                <select
                  id="gpo-topic-select"
                  className="nlds-select"
                  value={selectedTopic}
                  onChange={(e) => setSelectedTopic(e.target.value)}
                >
                  <option value="">Alle onderwerpen</option>
                  <option value="Toelaatbaarheidsverklaring (TLV)">
                    Toelaatbaarheidsverklaring (TLV)
                  </option>
                  <option value="Ontwikkelingsperspectief (OPP)">
                    Ontwikkelingsperspectief (OPP)
                  </option>
                  <option value="Verwijdering">Schorsing &amp; Verwijdering</option>
                  <option value="Toelating">Toelating &amp; Weigering</option>
                  <option value="Extra ondersteuning">Zorgplicht &amp; Extra ondersteuning</option>
                </select>
              </div>

              <div className="nlds-form-field">
                <label htmlFor="gpo-sector-select" className="nlds-label">
                  Onderwijssector (optioneel filter)
                </label>
                <select
                  id="gpo-sector-select"
                  className="nlds-select"
                  value={selectedSector}
                  onChange={(e) => setSelectedSector(e.target.value)}
                >
                  <option value="">Alle onderwijssectoren</option>
                  <option value="Primair onderwijs">Primair onderwijs (PO)</option>
                  <option value="Voortgezet onderwijs">Voortgezet onderwijs (VO)</option>
                  <option value="Speciaal onderwijs">Speciaal onderwijs (SO / VSO)</option>
                </select>
              </div>

              <div className="nlds-form-field">
                <label htmlFor="gpo-limit-select" className="nlds-label">
                  Aantal uitspraken tonen
                </label>
                <select
                  id="gpo-limit-select"
                  className="nlds-select"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                >
                  <option value={3}>Top 3 uitspraken</option>
                  <option value={5}>Top 5 uitspraken</option>
                  <option value={10}>Top 10 uitspraken</option>
                </select>
              </div>
            </div>

            <div className="nlds-search-actions">
              <Button
                type="submit"
                purpose="primary"
                disabled={loading}
              >
                {loading
                  ? 'Bezig met doorzoeken van Pinecone & Supabase…'
                  : 'Doorzoek GPO-uitspraken'}
              </Button>
              {question && (
                <Button
                  type="button"
                  purpose="secondary"
                  onClick={() => {
                    setQuestion('');
                    setSelectedTopic('');
                    setSelectedSector('');
                    setSearchResult(null);
                    setErrorMessage(null);
                  }}
                >
                  Wis zoekopdracht
                </Button>
              )}
            </div>
          </form>

          {/* Example questions for quick inspection */}
          <div className="nlds-examples" aria-label="Voorbeeldvragen">
            <span className="nlds-examples__label">Direct testen met een voorbeeldvraag:</span>
            <div className="nlds-examples__buttons">
              {EXAMPLE_QUESTIONS.map((item) => (
                <Button
                  key={item.label}
                  type="button"
                  purpose="subtle"
                  onClick={() => handleExampleClick(item.question, item.topic)}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
        </section>

        {/* B1 Error Alert */}
        {errorMessage && (
          <section
            className="nlds-alert nlds-alert--error"
            role="alert"
            aria-live="assertive"
          >
            <Heading level={3}>Let op: zoekopdracht kon niet worden uitgevoerd</Heading>
            <Paragraph>{errorMessage}</Paragraph>
          </section>
        )}

        {/* Clarification prompt when question is too vague */}
        {searchResult?.status === 'needs_clarification' && (
          <section
            className="nlds-alert nlds-alert--warning"
            role="status"
            aria-live="polite"
          >
            <Heading level={3}>Uw vraag heeft iets meer context nodig</Heading>
            <Paragraph>
              {searchResult.clarification_question ||
                searchResult.reason ||
                'Geef aan over welk onderwerp binnen passend onderwijs uw vraag gaat (bijvoorbeeld TLV, OPP, toelating of verwijdering).'}
            </Paragraph>
          </section>
        )}

        {/* Results Section */}
        {searchResult && searchResult.status !== 'needs_clarification' && (
          <section className="nlds-results" aria-labelledby="results-heading">
            <div className="nlds-results__header">
              <div>
                <Heading level={2} id="results-heading">
                  Gevonden GPO-uitspraken ({searchResult.results?.length ?? 0})
                </Heading>
                <Paragraph className="nlds-results__meta">
                  Zoekmethode:{' '}
                  <Code>{searchResult.method || 'pinecone_bge'}</Code> &bull; Index:{' '}
                  <Code>{searchResult.index_name || 'gpo-uitspraken'}</Code> &bull; Scope:{' '}
                  <span>{searchResult.scope}</span>
                </Paragraph>
              </div>
            </div>

            {!searchResult.results || searchResult.results.length === 0 ? (
              <div className="nlds-empty-state" role="status">
                <Heading level={3}>Geen uitspraken gevonden voor deze combinatie</Heading>
                <Paragraph>
                  {searchResult.reason ||
                    'Probeer een minder specifiek filter (zet Onderwerp of Onderwijssector op "Alle") of gebruik andere zoekwoorden zoals zorgplicht, ontwikkelingsperspectief, TLV of verwijdering.'}
                </Paragraph>
              </div>
            ) : (
              <div className="nlds-cases-list">
                {searchResult.results.map((caseItem, index) => (
                  <CaseCard
                    key={`${caseItem.case_number}-${caseItem.processing_id}`}
                    item={caseItem}
                    rank={index + 1}
                    onSelectCitation={handleSelectCitation}
                  />
                ))}
              </div>
            )}
          </section>
        )}
      </main>

      {/* Footer with transparency & standards info */}
      <footer className="nlds-footer" role="contentinfo">
        <div className="nlds-container nlds-footer__inner">
          <Paragraph>
            <strong>GPO-Assistent</strong> &bull; Gebouwd met componenten van het{' '}
            <Link href="https://nldesignsystem.nl" target="_blank" rel="noopener noreferrer">
              NL Design System (@nl-design-system)
            </Link>{' '}
            &bull; Brondata afkomstig van{' '}
            <Link
              href="https://www.onderwijsgeschillen.nl"
              target="_blank"
              rel="noopener noreferrer"
            >
              Stichting Onderwijsgeschillen
            </Link>
            .
          </Paragraph>
        </div>
      </footer>

      {/* Verifiable Source Citation Drawer ("Data bij de bron") */}
      <CitationDrawer
        citation={activeCitation}
        loading={citationLoading}
        error={citationError}
        onClose={() => {
          setActiveCitation(null);
          setCitationError(null);
        }}
      />
    </div>
  );
};
