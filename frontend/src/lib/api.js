import { API_BASE_URL, DEMO_MODE } from "./constants";
import { buildDemoAnswer } from "./mockData";

export async function sendChatMessage({ sessionId, content }) {
  if (DEMO_MODE) {
    return buildDemoAnswer(content);
  }

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: content })
  });

  if (!response.ok) {
    throw new Error(`Chat request failed with ${response.status}`);
  }

  return response.json();
}
