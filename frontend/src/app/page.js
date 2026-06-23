"use client";

import { useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { useChat } from "@/hooks/useChat";
import { useTheme } from "@/hooks/useTheme";

export default function HomePage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const {
    conversations,
    activeConversation,
    activeId,
    isStreaming,
    setActiveId,
    newConversation,
    sendMessage
  } = useChat();

  return (
    <main className="app-shell">
      <Header
        theme={theme}
        onToggleTheme={toggleTheme}
        onNewConversation={newConversation}
        onMenu={() => setSidebarOpen(true)}
      />
      <div className="workspace">
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={setActiveId}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />
        <ChatWindow
          conversation={activeConversation}
          isStreaming={isStreaming}
          onSendMessage={sendMessage}
        />
      </div>
    </main>
  );
}
