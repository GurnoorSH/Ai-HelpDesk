import { CheckCircle2, Loader2, Wrench } from "lucide-react";

export default function ToolCallCard({ call }) {
  const isComplete = call.status === "complete";

  return (
    <div className="tool-card">
      <div className="tool-icon" aria-hidden="true">
        {isComplete ? <CheckCircle2 size={17} /> : <Loader2 size={17} />}
      </div>
      <div>
        <span className="tool-title">
          <Wrench aria-hidden="true" size={14} />
          {call.label || call.name}
        </span>
        <p>{isComplete ? call.result : "Running lookup"}</p>
      </div>
    </div>
  );
}
