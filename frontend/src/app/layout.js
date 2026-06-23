import "./globals.css";

export const metadata = {
  title: "AI HelpDesk",
  description: "RAG-powered customer support workspace"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
