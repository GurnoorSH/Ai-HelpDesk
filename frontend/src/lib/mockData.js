const now = new Date("2026-06-23T10:20:00+05:30");

export const demoConversations = [
  {
    id: "conv-returns",
    title: "Return window",
    preview: "Can I return opened electronics?",
    updatedAt: new Date(now.getTime() - 8 * 60 * 1000).toISOString(),
    triage: { priority: "P3", category: "returns", confidence: 0.91 },
    messages: [
      {
        id: "m-1",
        role: "user",
        content: "Can I return opened electronics?",
        createdAt: new Date(now.getTime() - 10 * 60 * 1000).toISOString()
      },
      {
        id: "m-2",
        role: "assistant",
        content:
          "Electronics can be returned within 14 days when they are in like-new condition and include the original packaging, accessories, and receipt. Opened items may be inspected before approval.",
        createdAt: new Date(now.getTime() - 8 * 60 * 1000).toISOString(),
        sources: [
          {
            id: "src-1",
            title: "Store_Return_Policy.pdf",
            section: "Electronics returns",
            score: 0.92,
            excerpt:
              "Electronics are eligible for return within 14 days when returned with original packaging, accessories, and proof of purchase."
          }
        ],
        triage: { priority: "P3", category: "returns", confidence: 0.91 }
      }
    ]
  },
  {
    id: "conv-order",
    title: "Order ORD-001",
    preview: "Where is ORD-001?",
    updatedAt: new Date(now.getTime() - 36 * 60 * 1000).toISOString(),
    triage: { priority: "P4", category: "shipping", confidence: 0.88 },
    messages: [
      {
        id: "m-3",
        role: "user",
        content: "Can you check order ORD-001?",
        createdAt: new Date(now.getTime() - 38 * 60 * 1000).toISOString()
      },
      {
        id: "m-4",
        role: "assistant",
        content:
          "Order ORD-001 is currently in transit. The latest mock order status shows it shipped and is expected to arrive by June 25.",
        createdAt: new Date(now.getTime() - 36 * 60 * 1000).toISOString(),
        toolCalls: [
          {
            id: "tool-1",
            name: "check_order_status",
            status: "complete",
            label: "Order lookup",
            result: "ORD-001 - shipped - ETA Jun 25"
          }
        ],
        triage: { priority: "P4", category: "shipping", confidence: 0.88 }
      }
    ]
  },
  {
    id: "conv-billing",
    title: "Payment failed",
    preview: "My card was charged twice",
    updatedAt: new Date(now.getTime() - 80 * 60 * 1000).toISOString(),
    triage: {
      priority: "P2",
      category: "billing",
      confidence: 0.94,
      escalated: true
    },
    messages: [
      {
        id: "m-5",
        role: "user",
        content: "My card was charged twice and payment still failed.",
        createdAt: new Date(now.getTime() - 85 * 60 * 1000).toISOString()
      },
      {
        id: "m-6",
        role: "assistant",
        content:
          "I can help collect the details for a billing review. I will flag this for a support specialist because it involves a duplicate charge.",
        createdAt: new Date(now.getTime() - 80 * 60 * 1000).toISOString(),
        triage: {
          priority: "P2",
          category: "billing",
          confidence: 0.94,
          escalated: true
        }
      }
    ]
  }
];

export const demoTickets = [
  {
    id: "TCK-1042",
    customer: "Avery Stone",
    subject: "Duplicate charge after failed payment",
    priority: "P2",
    category: "billing",
    status: "Escalated",
    confidence: 0.94,
    updatedAt: new Date(now.getTime() - 80 * 60 * 1000).toISOString(),
    firstResponseMins: 2,
    cost: 0.034,
    summary:
      "Billing rule override triggered because the message mentions a failed payment and duplicate charge.",
    conversationId: "conv-billing"
  },
  {
    id: "TCK-1041",
    customer: "Mika Chen",
    subject: "Return window for opened electronics",
    priority: "P3",
    category: "returns",
    status: "AI Drafted",
    confidence: 0.91,
    updatedAt: new Date(now.getTime() - 8 * 60 * 1000).toISOString(),
    firstResponseMins: 1,
    cost: 0.027,
    summary:
      "Policy answer grounded in the return PDF with one high-confidence citation.",
    conversationId: "conv-returns"
  },
  {
    id: "TCK-1040",
    customer: "Noah Reed",
    subject: "Order ORD-001 delivery status",
    priority: "P4",
    category: "shipping",
    status: "Resolved",
    confidence: 0.88,
    updatedAt: new Date(now.getTime() - 36 * 60 * 1000).toISOString(),
    firstResponseMins: 1,
    cost: 0.012,
    summary:
      "Order lookup completed through the mock orders API and resolved without escalation.",
    conversationId: "conv-order"
  },
  {
    id: "TCK-1039",
    customer: "Priya Nair",
    subject: "Wrong size delivered",
    priority: "P3",
    category: "shipping",
    status: "Open",
    confidence: 0.82,
    updatedAt: new Date(now.getTime() - 3 * 60 * 60 * 1000).toISOString(),
    firstResponseMins: 4,
    cost: 0.021,
    summary:
      "Needs follow-up for order number before the shipping workflow can resolve it.",
    conversationId: "conv-size"
  }
];

export const dashboardMetrics = {
  operations: [
    { label: "AI deflection", value: "71%", delta: "+8%", tone: "good" },
    { label: "First contact", value: "64%", delta: "+5%", tone: "good" },
    { label: "p95 latency", value: "2.8s", delta: "-0.4s", tone: "good" },
    { label: "Error rate", value: "1.2%", delta: "-0.3%", tone: "good" }
  ],
  evaluation: [
    { label: "Faithfulness", value: 0.92 },
    { label: "Answer relevancy", value: 0.98 },
    { label: "Context recall", value: 0.86 },
    { label: "Hit@5", value: 0.93 }
  ],
  costs: {
    costPerTicket: 0.031,
    humanBaseline: 8,
    tokens: [
      { label: "Router", value: 1200 },
      { label: "HyDE", value: 3100 },
      { label: "Compression", value: 4700 },
      { label: "Final answer", value: 6500 },
      { label: "Critic", value: 2300 }
    ]
  },
  categories: [
    { label: "Returns", value: 42 },
    { label: "Shipping", value: 31 },
    { label: "Billing", value: 18 },
    { label: "Account", value: 9 }
  ]
};

export function buildDemoAnswer(message) {
  const text = message.toLowerCase();

  if (text.includes("ord-001") || text.includes("order")) {
    return {
      content:
        "Order ORD-001 is currently marked as shipped in the mock orders API. The expected delivery date is June 25, and the conversation is tagged as shipping with low urgency.",
      triage: { priority: "P4", category: "shipping", confidence: 0.89 },
      toolCalls: [
        {
          id: `tool-${Date.now()}`,
          name: "check_order_status",
          status: "complete",
          label: "Order lookup",
          result: "ORD-001 - shipped - ETA Jun 25"
        }
      ],
      sources: []
    };
  }

  if (
    text.includes("charge") ||
    text.includes("payment") ||
    text.includes("invoice")
  ) {
    return {
      content:
        "I can collect the billing details, but this should be escalated to a human specialist because it involves payment or charge handling. I have tagged it as high-priority billing.",
      triage: {
        priority: "P2",
        category: "billing",
        confidence: 0.94,
        escalated: true
      },
      toolCalls: [],
      sources: []
    };
  }

  return {
    content:
      "Based on the return policy, most items can be returned when they meet the stated return window and condition requirements. Electronics use the stricter electronics window, and the customer should include the receipt, packaging, and accessories.",
    triage: { priority: "P3", category: "returns", confidence: 0.9 },
    toolCalls: [],
    sources: [
      {
        id: `src-${Date.now()}`,
        title: "Store_Return_Policy.pdf",
        section: "Returns policy",
        score: 0.91,
        excerpt:
          "Returned items must meet the applicable window, condition, packaging, and proof-of-purchase requirements."
      }
    ]
  };
}
