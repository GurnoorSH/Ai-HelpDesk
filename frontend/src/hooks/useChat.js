"use client";

import { useMemo, useState } from "react";
import { sendChatMessage } from "@/lib/api";
import { demoConversations } from "@/lib/mockData";

function createMessage(role, content, extras = {}) {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    createdAt: new Date().toISOString(),
    ...extras
  };
}

function streamText(target, onChunk, onDone) {
  const words = target.split(" ");
  let index = 0;

  const interval = window.setInterval(() => {
    index += 1;
    onChunk(words.slice(0, index).join(" "));

    if (index >= words.length) {
      window.clearInterval(interval);
      onDone();
    }
  }, 28);
}

export function useChat() {
  const [conversations, setConversations] = useState(demoConversations);
  const [activeId, setActiveId] = useState(demoConversations[0].id);
  const [isStreaming, setIsStreaming] = useState(false);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeId),
    [activeId, conversations]
  );

  function updateConversation(conversationId, updater) {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === conversationId ? updater(conversation) : conversation
      )
    );
  }

  function newConversation() {
    const conversation = {
      id: `conv-${Date.now()}`,
      title: "New conversation",
      preview: "No messages yet",
      updatedAt: new Date().toISOString(),
      triage: { priority: "P4", category: "other", confidence: 0 },
      messages: []
    };

    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
  }

  async function sendMessage(content) {
    if (!content.trim() || isStreaming || !activeConversation) return;

    const cleanContent = content.trim();
    const userMessage = createMessage("user", cleanContent);
    const assistantMessage = createMessage("assistant", "", {
      isStreaming: true
    });

    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      title:
        conversation.messages.length === 0
          ? cleanContent.slice(0, 42)
          : conversation.title,
      preview: cleanContent,
      updatedAt: new Date().toISOString(),
      messages: [...conversation.messages, userMessage, assistantMessage]
    }));

    setIsStreaming(true);

    try {
      const response = await sendChatMessage({
        sessionId: activeConversation.id,
        content: cleanContent
      });

      streamText(
        response.content,
        (partial) => {
          updateConversation(activeConversation.id, (conversation) => ({
            ...conversation,
            messages: conversation.messages.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, content: partial }
                : message
            )
          }));
        },
        () => {
          updateConversation(activeConversation.id, (conversation) => ({
            ...conversation,
            preview: response.content,
            updatedAt: new Date().toISOString(),
            triage: response.triage || conversation.triage,
            messages: conversation.messages.map((message) =>
              message.id === assistantMessage.id
                ? {
                    ...message,
                    isStreaming: false,
                    sources: response.sources,
                    toolCalls: response.toolCalls,
                    triage: response.triage
                  }
                : message
            )
          }));
          setIsStreaming(false);
        }
      );
    } catch (error) {
      updateConversation(activeConversation.id, (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.id === assistantMessage.id
            ? {
                ...message,
                isStreaming: false,
                content:
                  "The chat service is not reachable right now. Check the backend API URL and try again."
              }
            : message
        )
      }));
      setIsStreaming(false);
    }
  }

  return {
    conversations,
    activeConversation,
    activeId,
    isStreaming,
    setActiveId,
    newConversation,
    sendMessage
  };
}
