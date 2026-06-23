export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

export const DEMO_MODE = process.env.NEXT_PUBLIC_DEMO_MODE !== "false";

export const PRIORITY_LABELS = {
  P1: "Critical",
  P2: "High",
  P3: "Normal",
  P4: "Low"
};

export const CATEGORY_LABELS = {
  billing: "Billing",
  shipping: "Shipping",
  returns: "Returns",
  product_defect: "Defect",
  account: "Account",
  other: "Other"
};
