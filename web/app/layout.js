export const metadata = {
  title: "SLM Playground — slm125m-live",
  description:
    "Chat and next-word prediction against a 125M legal/financial language model trained from scratch, plus any HuggingFace causal LM.",
};

import "./globals.css";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
