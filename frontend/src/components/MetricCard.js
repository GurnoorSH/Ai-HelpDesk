export default function MetricCard({ label, value, delta, tone }) {
  return (
    <section className={`metric-card ${tone || ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {delta ? <small>{delta}</small> : null}
    </section>
  );
}
