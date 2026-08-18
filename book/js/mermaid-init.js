// Mermaid, theme-aware. Material toggles data-md-color-scheme on <body>;
// re-render on scheme change so diagrams follow light/dark.
document.addEventListener("DOMContentLoaded", function () {
  if (typeof mermaid === "undefined") return;
  const dark = () =>
    document.body.getAttribute("data-md-color-scheme") === "slate";
  mermaid.initialize({
    startOnLoad: true,
    theme: dark() ? "dark" : "default",
    securityLevel: "loose",
    flowchart: { curve: "basis" },
  });
});
