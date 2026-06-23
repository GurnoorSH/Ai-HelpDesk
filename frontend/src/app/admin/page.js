"use client";

import { useMemo, useState } from "react";
import { BarChart3, CircleDollarSign, Clock, Filter, Gauge } from "lucide-react";
import Header from "@/components/Header";
import MetricCard from "@/components/MetricCard";
import TriageBadge from "@/components/TriageBadge";
import { useTheme } from "@/hooks/useTheme";
import { CATEGORY_LABELS } from "@/lib/constants";
import { currency, formatDateTime, formatPercent } from "@/lib/format";
import { dashboardMetrics, demoTickets } from "@/lib/mockData";

const statuses = ["All", "Open", "AI Drafted", "Escalated", "Resolved"];

export default function AdminPage() {
  const { theme, toggleTheme } = useTheme();
  const [status, setStatus] = useState("All");
  const [selectedId, setSelectedId] = useState(demoTickets[0].id);

  const tickets = useMemo(() => {
    if (status === "All") return demoTickets;
    return demoTickets.filter((ticket) => ticket.status === status);
  }, [status]);

  const selectedTicket =
    demoTickets.find((ticket) => ticket.id === selectedId) || demoTickets[0];

  return (
    <main className="app-shell">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <div className="admin-shell">
        <section className="admin-title">
          <div>
            <h1>Admin Dashboard</h1>
            <p>Tickets, RAG quality, latency, and cost attribution</p>
          </div>
          <div className="status-tabs" aria-label="Ticket status">
            {statuses.map((item) => (
              <button
                key={item}
                type="button"
                className={status === item ? "active" : ""}
                onClick={() => setStatus(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </section>

        <section className="metrics-grid" aria-label="Operational metrics">
          {dashboardMetrics.operations.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </section>

        <div className="admin-grid">
          <section className="panel ticket-queue">
            <div className="panel-heading">
              <span>
                <Filter aria-hidden="true" size={17} />
                Ticket queue
              </span>
              <small>{tickets.length} active</small>
            </div>

            <div className="ticket-table" role="table">
              <div className="ticket-row header" role="row">
                <span>Ticket</span>
                <span>Priority</span>
                <span>Status</span>
                <span>Updated</span>
              </div>
              {tickets.map((ticket) => (
                <button
                  key={ticket.id}
                  className={`ticket-row ${
                    selectedTicket.id === ticket.id ? "active" : ""
                  }`}
                  type="button"
                  role="row"
                  onClick={() => setSelectedId(ticket.id)}
                >
                  <span>
                    <strong>{ticket.id}</strong>
                    <small>{ticket.subject}</small>
                  </span>
                  <span>
                    <TriageBadge triage={ticket} compact />
                  </span>
                  <span>{ticket.status}</span>
                  <span>{formatDateTime(ticket.updatedAt)}</span>
                </button>
              ))}
            </div>
          </section>

          <aside className="panel ticket-detail">
            <div className="panel-heading">
              <span>{selectedTicket.id}</span>
              <TriageBadge triage={selectedTicket} />
            </div>
            <h2>{selectedTicket.subject}</h2>
            <p>{selectedTicket.summary}</p>
            <dl className="detail-list">
              <div>
                <dt>Customer</dt>
                <dd>{selectedTicket.customer}</dd>
              </div>
              <div>
                <dt>Category</dt>
                <dd>{CATEGORY_LABELS[selectedTicket.category]}</dd>
              </div>
              <div>
                <dt>Confidence</dt>
                <dd>{formatPercent(selectedTicket.confidence)}</dd>
              </div>
              <div>
                <dt>First response</dt>
                <dd>{selectedTicket.firstResponseMins}m</dd>
              </div>
              <div>
                <dt>AI cost</dt>
                <dd>{currency(selectedTicket.cost)}</dd>
              </div>
            </dl>
            <div className="detail-actions">
              <button type="button">Assign</button>
              <button type="button">Resolve</button>
              <button type="button">Re-triage</button>
            </div>
          </aside>

          <section className="panel quality-panel">
            <div className="panel-heading">
              <span>
                <Gauge aria-hidden="true" size={17} />
                Evaluation
              </span>
              <small>Reviewed set</small>
            </div>
            <div className="score-list">
              {dashboardMetrics.evaluation.map((metric) => (
                <div key={metric.label} className="score-row">
                  <span>{metric.label}</span>
                  <div className="score-track" aria-hidden="true">
                    <span style={{ width: `${Math.round(metric.value * 100)}%` }} />
                  </div>
                  <strong>{formatPercent(metric.value)}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel cost-panel">
            <div className="panel-heading">
              <span>
                <CircleDollarSign aria-hidden="true" size={17} />
                Cost
              </span>
              <small>Per resolved ticket</small>
            </div>
            <div className="cost-hero">
              <strong>{currency(dashboardMetrics.costs.costPerTicket)}</strong>
              <span>vs {currency(dashboardMetrics.costs.humanBaseline)}</span>
            </div>
            <div className="token-bars">
              {dashboardMetrics.costs.tokens.map((item) => (
                <div key={item.label}>
                  <span>{item.label}</span>
                  <div className="score-track" aria-hidden="true">
                    <span style={{ width: `${Math.min(item.value / 70, 100)}%` }} />
                  </div>
                  <small>{item.value.toLocaleString()}</small>
                </div>
              ))}
            </div>
          </section>

          <section className="panel category-panel">
            <div className="panel-heading">
              <span>
                <BarChart3 aria-hidden="true" size={17} />
                Categories
              </span>
              <small>Last 7 days</small>
            </div>
            <div className="category-bars">
              {dashboardMetrics.categories.map((item) => (
                <div key={item.label}>
                  <span>{item.label}</span>
                  <div className="category-meter" aria-hidden="true">
                    <span style={{ height: `${item.value * 2}%` }} />
                  </div>
                  <strong>{item.value}%</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel timeline-panel">
            <div className="panel-heading">
              <span>
                <Clock aria-hidden="true" size={17} />
                SLA
              </span>
              <small>Today</small>
            </div>
            <div className="sla-list">
              <span>Open P1 tickets <strong>0</strong></span>
              <span>Median response <strong>1.7m</strong></span>
              <span>Escalation rate <strong>14%</strong></span>
              <span>Draft acceptance <strong>82%</strong></span>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
