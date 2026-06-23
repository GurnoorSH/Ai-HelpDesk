"use client";

import { FileText } from "lucide-react";

export default function SourceCard({ source }) {
  return (
    <details className="source-card">
      <summary>
        <span>
          <FileText aria-hidden="true" size={15} />
          {source.title}
        </span>
        <strong>{Math.round(source.score * 100)}%</strong>
      </summary>
      <div className="source-meter" aria-hidden="true">
        <span style={{ width: `${Math.round(source.score * 100)}%` }} />
      </div>
      <p>{source.excerpt}</p>
      {source.section ? <small>{source.section}</small> : null}
    </details>
  );
}
