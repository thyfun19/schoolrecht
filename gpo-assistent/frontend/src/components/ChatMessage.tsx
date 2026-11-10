import React, { useState } from 'react';
import { Heading } from '@nl-design-system-candidate/heading-react';
import { Paragraph } from '@nl-design-system-candidate/paragraph-react';
import { Button } from '@nl-design-system-candidate/button-react';
import { DataBadge } from '@nl-design-system-candidate/data-badge-react';
import { Code } from '@nl-design-system-candidate/code-react';
import type { ChatMessageItem, ChatCitationRef } from '../types/chat';
import { CaseCard } from './CaseCard';

interface ChatMessageProps {
  message: ChatMessageItem;
  onSelectCitation: (processingId: string, sourceId: string) => void;
  onSuggestedReply?: (reply: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onSelectCitation,
  onSuggestedReply,
}) => {
  const [showCases, setShowCases] = useState(false);
  const isUser = message.sender === 'user';

  return (
    <div
      className={`nlds-chat-bubble-wrapper ${
        isUser ? 'nlds-chat-bubble-wrapper--user' : 'nlds-chat-bubble-wrapper--assistant'
      }`}
    >
      <div className={`nlds-chat-bubble ${isUser ? 'nlds-chat-bubble--user' : 'nlds-chat-bubble--assistant'}`}>
        <div className="nlds-chat-bubble__meta">
          <span className="nlds-chat-bubble__sender">
            {isUser ? '👤 U' : '⚖️ SchoolBot — GPO-Assistent'}
          </span>
          <span className="nlds-chat-bubble__time">{message.timestamp}</span>
          {!isUser && message.method && (
            <DataBadge>Pinecone BGE-M3 + Supabase</DataBadge>
          )}
        </div>

        <div className="nlds-chat-bubble__content">
          <Paragraph className="nlds-chat-bubble__text">{message.text}</Paragraph>

          {/* Clarification prompt & suggestions */}
          {message.needsClarification && message.clarificationPrompt && (
            <div className="nlds-clarification-box" role="status">
              <strong>Vraagverduidelijking:</strong>
              <Paragraph>{message.clarificationPrompt}</Paragraph>
              {message.suggestedReplies && message.suggestedReplies.length > 0 && (
                <div className="nlds-quick-replies" role="group" aria-label="Voorgestelde keuzes">
                  {message.suggestedReplies.map((reply) => (
                    <Button
                      key={reply}
                      purpose="secondary"
                      onClick={() => onSuggestedReply && onSuggestedReply(reply)}
                    >
                      {reply}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Citations list for the assistant message */}
          {message.citations && message.citations.length > 0 && (
            <div className="nlds-chat-citations" aria-label="Verifieerbare bronnen">
              <span className="nlds-chat-citations__title">Controleerbare broncitaten ("Data bij de bron"):</span>
              <div className="nlds-citation-buttons">
                {message.citations.map((c: ChatCitationRef, idx) => (
                  <button
                    key={`${c.sourceId}-${idx}`}
                    type="button"
                    className="nlds-source-chip"
                    onClick={() => onSelectCitation(c.processingId, c.sourceId)}
                    aria-label={`Open broncitaat uit ${c.label}`}
                  >
                    📄 {c.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Expandable Case Cards */}
          {message.results && message.results.length > 0 && (
            <div className="nlds-chat-expandable-cases">
              <Button
                purpose="subtle"
                onClick={() => setShowCases((prev) => !prev)}
                aria-expanded={showCases}
              >
                {showCases
                  ? `▲ Verberg de ${message.results.length} onderliggende GPO-uitspraken`
                  : `▼ Toon de ${message.results.length} gevonden GPO-uitspraken met details`}
              </Button>

              {showCases && (
                <div className="nlds-chat-cases-feed">
                  {message.results.map((c, idx) => (
                    <CaseCard
                      key={`${c.case_number}-${c.processing_id}-${idx}`}
                      item={c}
                      rank={idx + 1}
                      onSelectCitation={onSelectCitation}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
