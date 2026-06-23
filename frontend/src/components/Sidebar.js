"use client";

import { Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { formatDateTime } from "@/lib/format";
import TriageBadge from "./TriageBadge";

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  open,
  onClose
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const search = query.trim().toLowerCase();
    if (!search) return conversations;

    return conversations.filter((conversation) =>
      [conversation.title, conversation.preview]
        .join(" ")
        .toLowerCase()
        .includes(search)
    );
  }, [conversations, query]);

  return (
    <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Conversations">
      <div className="sidebar-title">
        <h2>Conversations</h2>
        <button
          className="icon-button mobile-only"
          type="button"
          onClick={onClose}
          aria-label="Close conversations"
          title="Close"
        >
          <X size={18} />
        </button>
      </div>

      <label className="search-box">
        <Search aria-hidden="true" size={16} />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search"
        />
      </label>

      <div className="conversation-list">
        {filtered.map((conversation) => (
          <button
            key={conversation.id}
            type="button"
            className={`conversation-item ${
              conversation.id === activeId ? "active" : ""
            }`}
            onClick={() => {
              onSelect(conversation.id);
              onClose?.();
            }}
          >
            <span className="conversation-row">
              <strong>{conversation.title}</strong>
              <small>{formatDateTime(conversation.updatedAt)}</small>
            </span>
            <span className="conversation-preview">{conversation.preview}</span>
            <span className="conversation-row">
              <TriageBadge triage={conversation.triage} compact />
              <small>
                {Math.round((conversation.triage?.confidence || 0) * 100)}%
              </small>
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}
