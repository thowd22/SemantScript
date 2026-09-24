import type { ReactNode } from "react";

export const metadata = { title: "Ticket triage" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
