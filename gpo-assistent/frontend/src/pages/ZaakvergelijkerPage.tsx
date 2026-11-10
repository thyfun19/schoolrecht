import React, { useEffect, useState } from 'react';
import { Heading } from '@nl-design-system-candidate/heading-react';
import { Paragraph } from '@nl-design-system-candidate/paragraph-react';
import { Button } from '@nl-design-system-candidate/button-react';
import { DataBadge } from '@nl-design-system-candidate/data-badge-react';
import { SkipLink } from '@nl-design-system-candidate/skip-link-react';
import { fetchCitation, fetchHealth, searchPinecone } from '../api/client';
import type { CitationResponse, HealthResponse, PineconeCaseResult, PineconeSearchResponse } from '../types/gpo';
import { CaseCard } from '../components/CaseCard';
import { CitationDrawer } from '../components/CitationDrawer';

const EXAMPLE_QUESTIONS = [
  {
    category: 'Toelating & Zorgplicht',
    title: 'Onderzoek naar extra ondersteuning bij aanmelding',
    query: 'De school zegt dat onze dochter te veel hulp nodig heeft om te worden toegelaten. Hoe onderzocht de school wat ze zelf kon bieden?',
  },
  {
    category: 'Ontwikkelingsperspectief (OPP)',
    title: 'Inbreng en verbeterpunten van ouders op het plan',
    query: 'Wij stuurden verbeterpunten op het conceptplan. Wat deed de school volgens de commissie met onze reactie?',
  },
  {
    category: 'Schorsing & Verwijdering',
    title: 'Hoorplicht en verhaal doen door ouders',
    query: 'Onze zoon mag na een gesprek op school ineens niet meer op school komen. Op welk moment hadden de ouders hun verhaal mogen doen?',
  },
  {
    category: 'Thuiszitten & Terugkeer',
    title: 'Terugkeer naar school met een opbouwschema',
    query: "Na weken thuiszitten bespreekt school of onze dochter met een aangepast rooster terug kan komen. Hoe keek de commissie naar zo'n opbouw?",
  },
];

export const ZaakvergelijkerPage: React.FC = () => {
  const [searchInput, setSearchInput] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const [searchResponse, setSearchResponse] = useState<PineconeSearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(3);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const [activeCitation, setActiveCitation] = useState<CitationResponse | null>(null);
  const [citationLoading, setCitationLoading] = useState(false);
  const [citationError, setCitationError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const executeSearch = async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed) return;

    setLoading(true);
    setHasSearched(true);
    setSearchError(null);
    setActiveQuery(trimmed);
    setVisibleCount(3);

    try {
      const response = await searchPinecone({ question: trimmed, limit: 9 });
      setSearchResponse(response);
    } catch (err) {
      setSearchResponse(null);
      setSearchError(err instanceof Error ? err.message : 'Er is een fout opgetreden bij het doorzoeken van de uitspraken.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void executeSearch(searchInput);
  };

  const handleExampleClick = (exampleQuery: string) => {
    setSearchInput(exampleQuery);
    void executeSearch(exampleQuery);
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
        err instanceof Error ? err.message : 'Broncitaat kon niet worden opgehaald.'
      );
    } finally {
      setCitationLoading(false);
    }
  };

  const allResults: PineconeCaseResult[] = searchResponse?.results || [];
  const currentVisibleCases = allResults.slice(0, visibleCount);

  return (
    <div className="vr-page">
      <SkipLink href="#main-content">Direct naar hoofdinhoud</SkipLink>

      {/* Top Navbar */}
      <header className="vr-navbar" role="banner">
        <div className="vr-container vr-navbar__inner">
          <a href="/" className="vr-navbar__brand" aria-label="SchoolRecht Home">
            <img
              src="/schoolrecht-emblem.png"
              alt=""
              aria-hidden="true"
              className="vr-navbar__emblem"
            />
            <div className="vr-navbar__brand-group">
              <span className="vr-navbar__brand-text">SchoolRecht</span>
              <span className="vr-navbar__brand-sub">Passend Onderwijs</span>
            </div>
          </a>
          <div className="vr-navbar__stats">
            <DataBadge>
              {health ? `${health.counts.documents} uitspraken in bronregister` : '721 uitspraken in bronregister'}
            </DataBadge>
          </div>
        </div>
      </header>

      {/* SchoolRecht Hero met zwevend zoekpaneel */}
      <section className="vr-hero" aria-labelledby="hero-title">
        <div className="vr-container vr-hero__inner">
          <div className="vr-hero__kicker">Uitsprakenvergelijker voor ouders &amp; leerlingen</div>
          <Heading level={1} id="hero-title" className="vr-hero__title">
            Hoe oordeelt de Geschillencommissie in vergelijkbare schoolzaken?
          </Heading>
          <Paragraph className="vr-hero__subtitle">
            Beschrijf uw situatie in uw eigen woorden en bekijk direct de relevante overwegingen en paginabronnen uit eerdere uitspraken.
          </Paragraph>

          <div className="sr-search-panel">
            <form onSubmit={handleSubmit} className="vr-hero__search-form" noValidate>
              <input
                type="text"
                className="vr-hero__search-input"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Bijv. De school zegt dat de klas vol zit, of: hoe moet het OPP met ouders worden besproken?"
                aria-label="Beschrijf uw situatie of stel uw vraag"
                disabled={loading}
              />
              <Button
                type="submit"
                purpose="primary"
                className="vr-hero__btn-compare"
                disabled={loading || !searchInput.trim()}
              >
                {loading ? 'Zoeken…' : 'Zoek uitspraken'}
              </Button>
            </form>
            <div className="sr-search-panel__meta">
              <span>✓ Letterlijke citaten uit commissieoverwegingen</span>
              <span>✓ Directe PDF-paginaverwijzing</span>
              <span>✓ Informatief hulpmiddel (geen juridisch advies)</span>
            </div>
          </div>
        </div>
      </section>

      {/* Hoofdsectie: Resultaten OF Voorbeeldvragen */}
      <main id="main-content" className="vr-container" role="main">
        {!hasSearched && (
          <section className="vr-examples-container" aria-label="Veelgestelde oudervragen">
            <div className="vr-examples-header">
              <Heading level={2} className="vr-examples-title">
                Veelvoorkomende situaties van ouders
              </Heading>
              <Paragraph className="vr-examples-subtitle">
                Klik op een voorbeeldsituatie om direct te zien hoe de commissie vergelijkbare zaken beoordeelde:
              </Paragraph>
            </div>
            <div className="vr-examples-grid">
              {EXAMPLE_QUESTIONS.map((item) => (
                <div
                  key={item.title}
                  className="vr-example-card"
                  role="button"
                  tabIndex={0}
                  onClick={() => handleExampleClick(item.query)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleExampleClick(item.query);
                    }
                  }}
                >
                  <div className="vr-example-card__badge">
                    <DataBadge>{item.category}</DataBadge>
                  </div>
                  <Heading level={3} className="vr-example-card__heading">
                    {item.title}
                  </Heading>
                  <Paragraph className="vr-example-card__text">“{item.query}”</Paragraph>
                  <span className="vr-example-card__cta">Bekijk vergelijkbare uitspraken →</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Na het zoeken: Resultatenoverzicht */}
        {hasSearched && (
          <section className="vr-results-wrapper" aria-label="Zoekresultaten">
            {loading && (
              <div className="sr-loading-card">
                <Paragraph style={{ fontSize: '18px', fontWeight: 600, color: '#231f6b', margin: 0 }}>
                  Bezig met het zoeken naar relevante commissieoverwegingen en paginabronnen…
                </Paragraph>
              </div>
            )}

            {searchError && (
              <div className="sr-error-card">
                <strong>Toelichting:</strong> {searchError}
              </div>
            )}

            {!loading && !searchError && allResults.length > 0 && (
              <>
                <div className="sr-results-toolbar">
                  <div>
                    <Heading level={2} className="vr-results-counter" style={{ margin: 0 }}>
                      {allResults.length} {allResults.length === 1 ? 'vergelijkbare uitspraak' : 'vergelijkbare uitspraken'}
                    </Heading>
                    <Paragraph className="sr-results-query">
                      Geselecteerd op basis van uw vraag: <em>“{activeQuery}”</em>
                    </Paragraph>
                  </div>
                  <Button
                    purpose="subtle"
                    className="vr-link-button"
                    onClick={() => {
                      setHasSearched(false);
                      setSearchInput('');
                      setSearchResponse(null);
                      setSearchError(null);
                    }}
                  >
                    ← Andere vraag stellen
                  </Button>
                </div>

                <div className="vr-cases-list">
                  {currentVisibleCases.map((caseItem, idx) => (
                    <CaseCard
                      key={`${caseItem.case_number}-${caseItem.processing_id}`}
                      item={caseItem}
                      rank={idx + 1}
                      searchQuery={activeQuery}
                      onSelectCitation={handleSelectCitation}
                    />
                  ))}
                </div>

                {visibleCount < allResults.length && (
                  <Button
                    type="button"
                    purpose="primary"
                    className="vr-btn-load-more"
                    onClick={() => setVisibleCount((prev) => prev + 3)}
                  >
                    Toon nog {Math.min(3, allResults.length - visibleCount)} uitspraken ({visibleCount} van {allResults.length} zichtbaar)
                  </Button>
                )}
              </>
            )}

            {!loading && !searchError && allResults.length === 0 && (
              <>
                <div className="vr-no-results">
                  <Paragraph className="vr-no-results__text">Geen passende uitspraken gevonden voor deze zoekopdracht</Paragraph>
                </div>

                <section className="vr-examples-container" aria-label="Veelgestelde oudervragen">
                  <Heading level={2} className="vr-examples-title">
                    Probeer een van deze voorbeeldsituaties:
                  </Heading>
                  <div className="vr-examples-grid">
                    {EXAMPLE_QUESTIONS.map((item) => (
                      <div
                        key={item.title}
                        className="vr-example-card"
                        role="button"
                        tabIndex={0}
                        onClick={() => handleExampleClick(item.query)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            handleExampleClick(item.query);
                          }
                        }}
                      >
                        <div className="vr-example-card__badge">
                          <DataBadge>{item.category}</DataBadge>
                        </div>
                        <Heading level={3} className="vr-example-card__heading">
                          {item.title}
                        </Heading>
                        <Paragraph className="vr-example-card__text">“{item.query}”</Paragraph>
                      </div>
                    ))}
                  </div>
                </section>
              </>
            )}
          </section>
        )}
      </main>

      {/* Disclaimer en verantwoording specifiek voor SchoolRecht & Passend Onderwijs */}
      <section className="vr-disclaimer-section" aria-label="Disclaimer en juridische verantwoording">
        <div className="vr-container">
          <div className="vr-disclaimer-box">
            <Heading level={3} className="vr-disclaimer-box__title">
              Over dit onderzoeksprototype &amp; Juridische Verantwoording
            </Heading>
            <Paragraph className="vr-disclaimer-box__text">
              SchoolRecht is een academisch onderzoeks- en onderwijsprototype ontwikkeld binnen Studio Public Lab aan de
              Hogeschool van Amsterdam (HvA). Het doel is om openbare uitspraken van de Geschillencommissie Passend Onderwijs
              (GPO) toegankelijk, doorzoekbaar en vergelijkbaar te maken voor ouders met behulp van taaltechnologie en directe bronverantwoording.
            </Paragraph>
            <Paragraph className="vr-disclaimer-box__text">
              Deze applicatie biedt uitsluitend algemene, informatieve ondersteuning en vormt uitdrukkelijk <strong>geen juridisch advies</strong>.
              Aan de getoonde citaten of vergelijkingen kunnen geen rechten worden ontleend. Raadpleeg voor formele
              besluitvorming of termijnen altijd de officiële publicaties op <em>Onderwijsgeschillen.nl</em> of een
              gekwalificeerd onderwijsjurist.
            </Paragraph>
          </div>
        </div>
      </section>

      {/* Subtiele footer */}
      <footer className="vr-footer-bottom" role="contentinfo">
        <div className="vr-container vr-footer-bottom__inner">
          <Paragraph style={{ margin: 0 }}>
            &copy; {new Date().getFullYear()} SchoolRecht &bull; Onderwijsprototype Hogeschool van Amsterdam (Studio Public Lab)
          </Paragraph>
        </div>
      </footer>

      {/* Geverifieerde bronpassage drawer */}
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
