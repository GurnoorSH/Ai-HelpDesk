"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, MessageSquarePlus, PanelsTopLeft } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

export default function Header({
  theme,
  onToggleTheme,
  onNewConversation,
  onMenu
}) {
  const pathname = usePathname();
  const isAdmin = pathname?.startsWith("/admin");

  return (
    <header className="app-header">
      <div className="header-left">
        {onMenu ? (
          <button
            className="icon-button mobile-only"
            type="button"
            onClick={onMenu}
            aria-label="Open conversations"
            title="Conversations"
          >
            <Menu size={18} />
          </button>
        ) : null}
        <Link className="brand" href="/">
          <span className="brand-mark">AI</span>
          <span>
            <strong>AI HelpDesk</strong>
            <small>RAG support workspace</small>
          </span>
        </Link>
      </div>

      <nav className="top-nav" aria-label="Primary navigation">
        <Link className={!isAdmin ? "active" : ""} href="/">
          Chat
        </Link>
        <Link className={isAdmin ? "active" : ""} href="/admin">
          Admin
        </Link>
      </nav>

      <div className="header-actions">
        {onNewConversation ? (
          <button
            className="action-button"
            type="button"
            onClick={onNewConversation}
          >
            <MessageSquarePlus size={17} />
            <span>New</span>
          </button>
        ) : (
          <Link className="action-button" href="/">
            <PanelsTopLeft size={17} />
            <span>Chat</span>
          </Link>
        )}
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  );
}
