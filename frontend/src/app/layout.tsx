import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PyRete ReAct Hardware Pricing Agent",
  description: "Goal-driven forward chaining rule coordinator for hardware pricing & international conversions",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
