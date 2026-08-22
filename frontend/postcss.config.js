/**
 * PostCSS config — with the Tailwind design tokens inlined directly as the
 * plugin's options object, instead of a separate tailwind.config.ts file.
 * (Tailwind's PostCSS plugin accepts either a path to a config file or the
 * config object itself.)
 *
 * "Civil register" design tokens: paper is deliberately warmer/more ochre
 * than the generic AI-cream (#F4F1EA) so it reads as aged ledger paper,
 * not a template default.
 */
const tailwindConfig = {
  content: ["./app/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#14213D",           // deep navy — masthead, headings, primary text
        "ink-faint": "#5C6478",
        paper: "#EDE6D3",         // aged ledger paper background
        "paper-line": "#D8CDB0",  // hairline rule on paper
        marigold: "#E8912D",      // civic/petition accent — links, eyebrows
        seal: "#A23B2E",          // rubber-stamp red — negative sentiment
        moss: "#4B6B4E",          // positive sentiment
        slate: "#6B7280",         // neutral sentiment
      },
      fontFamily: {
        display: ["Fraunces", "ui-serif", "Georgia", "serif"],
        body: ["'IBM Plex Sans'", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      backgroundImage: {
        "paper-fiber":
          "radial-gradient(circle at 20% 20%, rgba(20,33,61,0.03) 0, transparent 45%), radial-gradient(circle at 80% 60%, rgba(20,33,61,0.025) 0, transparent 40%)",
      },
    },
  },
  plugins: [],
};

module.exports = {
  plugins: {
    tailwindcss: tailwindConfig,
    autoprefixer: {},
  },
};
