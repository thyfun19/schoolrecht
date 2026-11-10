import React, { useEffect, useRef, useState } from 'react';
import { Heading } from '@nl-design-system-candidate/heading-react';
import { Paragraph } from '@nl-design-system-candidate/paragraph-react';
import { Button } from '@nl-design-system-candidate/button-react';
import { Link } from '@nl-design-system-candidate/link-react';
import { DataBadge } from '@nl-design-system-candidate/data-badge-react';
import { SkipLink } from '@nl-design-system-candidate/skip-link-react';
import { fetchCitation, fetchHealth, searchPinecone } from '../api/client';
import type { CitationResponse, HealthResponse } from '../types/gpo';
import type { ChatMessageItem } from '../types/chat';
import { ChatMessage } from '../components/ChatMessage';
import { CitationDrawer } from '../components/CitationDrawer';
import { synthesizeAnswer } from '../utils/answerSynthesizer';

const STARTER_PROMPTS = [
  'Wanneer is een TLV onzorgvuldig voorbereid wegens ontbrekende deskundigen?',
  'Welke eisen stelt de commissie aan het ontwikkelingsperspectief (OPP) en OOGO met ouders?',
  'Mag een school een leerling met extra ondersteuningsbehoefte direct schorsen of verwijderen?',
  'Wanneer mag een school een leerling weigeren wegens handelingsverlegenheid?',
];

export const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessageItem[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      timestamp: 'Zojuist',
      text: 'Welkom bij SchoolBot! Ik ben uw AI-assistent voor de jurisprudentie van de Geschillencommissie Passend Onderwijs (GPO). U kunt mij een juridische vraag of casus voorleggen. Ik doorzoek direct 721 officiële uitspraken in Pinecone en Supabase en beantwoord uw vraag in begrijpelijke taal met verifieerbare broncitaten.',
      suggestedReplies: STARTER_PROMPTS,
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const [activeCitation, setActiveCitation] = useState<CitationResponse | null>(null);
  const [citationLoading, setCitationLoading] = useState(false);
  const [citationError, setCitationError] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSendMessage = async (textToSend: string) => {
    const trimmed = textToSend.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessageItem = {
      id: `user-${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: trimmed,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setLoading(true);

    try {
      const response = await searchPinecone({ question: trimmed, limit: 5 });

      if (response.status === 'needs_clarification') {
        const assistantMsg: ChatMessageItem = {
          id: `asst-${Date.now()}`,
          sender: 'assistant',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: 'Uw vraag kan specifieker worden beantwoord met iets meer context.',
          needsClarification: true,
          clarificationPrompt:
            response.clarification_question ||
            response.reason ||
            'Kies een onderwerp of geef aan om welk aspect van passend onderwijs het gaat.',
          suggestedReplies: [
            'Toelaatbaarheidsverklaring (TLV)',
            'Ontwikkelingsperspectief (OPP)',
            'Verwijdering van school',
            'Toelating en weigering',
          ],
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } else {
        const synth = synthesizeAnswer(trimmed, response);
        let fullText = synth.summary + '\n\n';
        synth.points.forEach((p, idx) => {
          fullText += `${idx + 1}. ${p.text}\n`;
        });
        fullText += `\n${synth.closing}`;

        const assistantMsg: ChatMessageItem = {
          id: `asst-${Date.now()}`,
          sender: 'assistant',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          text: fullText,
          results: response.results,
          citations: synth.citations,
          method: response.method || 'pinecone_bge',
        };
        setMessages((prev) => [...prev, assistantMsg]);
      }
    } catch (err) {
      const errorMsg: ChatMessageItem = {
        id: `asst-err-${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        text: `Sorry, er ging iets mis bij het ophalen van de uitspraken: ${
          err instanceof Error ? err.message : 'Onbekende fout'
        }. Probeer het opnieuw of herformuleer uw vraag.`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void handleSendMessage(inputText);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSendMessage(inputText);
    }
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

  const handleClearChat = () => {
    setMessages([
      {
        id: 'welcome-reset',
        sender: 'assistant',
        timestamp: 'Zojuist',
        text: 'Het gesprek is gewist. Stel gerust een nieuwe vraag over passend onderwijs!',
        suggestedReplies: STARTER_PROMPTS,
      },
    ]);
  };

  return (
    <div className="nlds-page nlds-chat-page-layout">
      <SkipLink href="#chat-main">Direct naar chatberichten</SkipLink>

      {/* Top Header */}
      <header className="nlds-header" role="banner">
        <div className="nlds-container nlds-header__inner">
          <div className="nlds-header__brand">
            <span className="nlds-header__Crest" aria-hidden="true">
              🤖
            </span>
            <div>
              <Heading level={1} className="nlds-header__title">
                SchoolBot — GPO Chatassistent
              </Heading>
              <Paragraph className="nlds-header__subtitle">
                Juridisch advies op basis van 721 officiële uitspraken (&ldquo;Data bij de bron&rdquo; &bull; WCAG 2.1 AA)
              </Paragraph>
            </div>
          </div>

          <div className="nlds-header__status" aria-live="polite">
            {health ? (
              <div className="nlds-status-badges">
                <DataBadge className="nlds-badge nlds-badge--online">
                  Supabase: {health.counts.documents} zaken / {health.counts.citations} citaten
                </DataBadge>
                <DataBadge className="nlds-badge nlds-badge--pinecone">
                  Pinecone BGE-M3 (1024d)
                </DataBadge>
              </div>
            ) : (
              <DataBadge className="nlds-badge nlds-badge--neutral">
                Verbinding controleren…
              </DataBadge>
            )}
          </div>
        </div>
      </header>

      {/* Main Chat Interface */}
      <main id="chat-main" className="nlds-container nlds-chat-main-wrapper" role="main">
        {/* Quick starter bar */}
        <section className="nlds-quick-starter-bar" aria-label="Snelle voorbeeldvragen">
          <span className="nlds-quick-starter-bar__label">Snel testen:</span>
          <div className="nlds-quick-starter-bar__buttons">
            {STARTER_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="nlds-starter-chip"
                onClick={() => void handleSendMessage(prompt)}
                disabled={loading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </section>

        {/* Message Feed */}
        <section
          className="nlds-chat-feed"
          aria-label="Chatgesprek met SchoolBot"
          aria-live="polite"
        >
          {messages.map((m) => (
            <ChatMessage
              key={m.id}
              message={m}
              onSelectCitation={handleSelectCitation}
              onSuggestedReply={(reply) => void handleSendMessage(reply)}
            />
          ))}

          {loading && (
            <div className="nlds-chat-bubble-wrapper nlds-chat-bubble-wrapper--assistant">
              <div className="nlds-chat-bubble nlds-chat-bubble--assistant nlds-chat-bubble--loading">
                <span className="nlds-chat-bubble__sender">⚖️ SchoolBot zoekt...</span>
                <Paragraph>
                  Bezig met semantisch zoeken in 721 uitspraken en 33.614 citaten...
                </Paragraph>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </section>

        {/* Chat Input Bar */}
        <footer className="nlds-chat-input-panel">
          <form onSubmit={handleSubmit} className="nlds-chat-form">
            <div className="nlds-chat-input-field">
              <label htmlFor="schoolbot-input" className="sr-only">
                Typ uw vraag aan SchoolBot
              </label>
              <textarea
                id="schoolbot-input"
                className="nlds-textarea nlds-chat-textarea"
                rows={2}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Stel een vraag over passend onderwijs (bijv. over TLV, OPP, zorgplicht of verwijdering)..."
                disabled={loading}
              />
            </div>
            <div className="nlds-chat-input-actions">
              <Button type="submit" purpose="primary" disabled={loading || !inputText.trim()}>
                {loading ? 'Zoeken...' : 'Verstuur vraag ➔'}
              </Button>
              <Button
                type="button"
                purpose="subtle"
                onClick={handleClearChat}
                disabled={loading}
              >
                Wis gesprek
              </Button>
            </div>
          </form>
          <span className="nlds-chat-footnote">
            Tip: Druk op <kbd>Enter</kbd> om te versturen, of <kbd>Shift+Enter</kbd> voor een nieuwe regel.
          </span>
        </footer>
      </main>

      {/* Verifiable Source Citation Drawer */}
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
