import { AlertTriangle, CircleDot, ShieldCheck, Zap } from "lucide-react";
import { CATEGORY_LABELS, PRIORITY_LABELS } from "@/lib/constants";

const priorityIcons = {
  P1: Zap,
  P2: AlertTriangle,
  P3: ShieldCheck,
  P4: CircleDot
};

export default function TriageBadge({ triage, compact = false }) {
  if (!triage) return null;

  const Icon = priorityIcons[triage.priority] || CircleDot;
  const priority = PRIORITY_LABELS[triage.priority] || triage.priority;
  const category = CATEGORY_LABELS[triage.category] || triage.category;

  return (
    <span className={`triage triage-${triage.priority || "P4"}`}>
      <Icon aria-hidden="true" size={14} />
      <span>{compact ? triage.priority : `${triage.priority} ${priority}`}</span>
      {!compact && category ? <span className="triage-category">{category}</span> : null}
    </span>
  );
}
