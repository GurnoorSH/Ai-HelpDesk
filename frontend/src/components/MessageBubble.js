import { Bot, UserRound } from "lucide-react";
import { formatTime } from "@/lib/format";
import SourceCard from "./SourceCard";
import StreamingText from "./StreamingText";
import ToolCallCard from "./ToolCallCard";
import TriageBadge from "./TriageBadge";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <article className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? <UserRound size={17} /> : <Bot size={17} />}
      </div>
      <div className="message-body">
        <div className="message-meta">
          <strong>{isUser ? "Customer" : "AI Agent"}</strong>
          <span>{formatTime(message.createdAt)}</span>
          {!isUser ? <TriageBadge triage={message.triage} compact /> : null}
        </div>
        <p>
          <StreamingText active={message.isStreaming}>
            {message.content}
          </StreamingText>
        </p>

        {message.toolCalls?.length ? (
          <div className="message-stack">
            {message.toolCalls.map((call) => (
              <ToolCallCard key={call.id} call={call} />
            ))}
          </div>
        ) : null}

        {message.sources?.length ? (
          <div className="source-list">
            {message.sources.map((source) => (
              <SourceCard key={source.id} source={source} />
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}
