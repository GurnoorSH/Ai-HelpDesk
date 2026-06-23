"use client";

import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";
import TriageBadge from "./TriageBadge";

export default function ChatWindow({
  conversation,
  isStreaming,
  onSendMessage
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [conversation?.messages]);

  if (!conversation) {
    return (
      <section className="chat-panel empty">
        <Sparkles size={24} />
        <p>Select a conversation</p>
      </section>
    );
  }

  return (
    <section className="chat-panel">
      <div className="chat-context">
        <div>
          <h1>{conversation.title}</h1>
          <p>{conversation.preview}</p>
        </div>
        <TriageBadge triage={conversation.triage} />
      </div>

      <div className="messages" aria-live="polite">
        {conversation.messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={onSendMessage} disabled={isStreaming} />
    </section>
  );
}
