import "./globals.css";

export const metadata = {
  title: "JanMat — Public Consultation Register",
  description:
    "A plain-language snapshot of public feedback on tracked bills, built from PRS Legislative Research stakeholder analysis.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        {/* Display serif (masthead/titles), body sans, mono for data/labels */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
