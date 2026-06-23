"use client";

import { SendHorizontal } from "lucide-react";
import { useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");

  function submit() {
    const cleanValue = value.trim();
    if (!cleanValue || disabled) return;
    onSend(cleanValue);
    setValue("");
  }

  return (
    <form
      className="chat-input"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
        placeholder="Message the support agent"
        rows={1}
        disabled={disabled}
      />
      <button
        className="send-button"
        type="submit"
        disabled={disabled || !value.trim()}
        aria-label="Send message"
        title="Send"
      >
        <SendHorizontal size={19} />
      </button>
    </form>
  );
}
