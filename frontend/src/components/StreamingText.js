export default function StreamingText({ children, active }) {
  return (
    <span>
      {children}
      {active ? <span className="stream-cursor" aria-hidden="true" /> : null}
    </span>
  );
}
