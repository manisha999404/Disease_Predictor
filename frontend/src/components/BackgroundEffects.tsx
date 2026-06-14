import { useMemo } from "react";

export function BackgroundEffects() {
  // All values computed once — stable across renders, no useState/useEffect needed
  const stars = useMemo(
    () =>
      Array.from({ length: 90 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        s: Math.random() * 2 + 0.5,
        d: Math.random() * 5,
      })),
    []
  );

  const orbs = useMemo(
    () =>
      Array.from({ length: 6 }, (_, i) => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: 200 + Math.random() * 300,
        color: ["#8B5CF6", "#60A5FA", "#22D3EE"][i % 3],
        d: Math.random() * 10,
      })),
    []
  );

  const lines = useMemo(
    () =>
      Array.from({ length: 8 }, () => ({
        x1: `${Math.random() * 100}%`,
        y1: `${Math.random() * 100}%`,
        x2: `${Math.random() * 100}%`,
        y2: `${Math.random() * 100}%`,
      })),
    []
  );

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      {/* Grid */}
      <div
        className="absolute inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(139,92,246,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.5) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
          maskImage: "radial-gradient(ellipse 80% 60% at 50% 40%, #000 30%, transparent 80%)",
        }}
      />

      {/* Aurora orbs — NO filter:blur (that creates compositing layers and freezes on focus).
          Instead use a radial-gradient that fades to transparent naturally.
          Visually identical, zero compositing cost. */}
      {orbs.map((o, i) => (
        <div
          key={i}
          className="absolute rounded-full animate-drift"
          style={{
            left: `${o.x}%`,
            top: `${o.y}%`,
            width: o.size,
            height: o.size,
            // Soft fade via gradient — no filter needed, no GPU layer created
            background: `radial-gradient(circle, ${o.color}40 0%, ${o.color}18 35%, transparent 70%)`,
            animationDelay: `${o.d}s`,
            // Isolate animation to its own layer without blur overhead
            willChange: "transform",
          }}
        />
      ))}

      {/* Stars */}
      {stars.map((s, i) => (
        <div
          key={i}
          className="absolute rounded-full bg-white animate-twinkle"
          style={{
            left: `${s.x}%`,
            top: `${s.y}%`,
            width: s.s,
            height: s.s,
            animationDelay: `${s.d}s`,
            boxShadow: "0 0 4px rgba(255,255,255,0.6)",
          }}
        />
      ))}

      {/* Neural lines */}
      <svg className="absolute inset-0 w-full h-full opacity-20">
        <defs>
          <linearGradient id="nl" x1="0" x2="1">
            <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0" />
            <stop offset="50%" stopColor="#22D3EE" />
            <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0" />
          </linearGradient>
        </defs>
        {lines.map((l, i) => (
          <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="url(#nl)" strokeWidth="0.5" />
        ))}
      </svg>
    </div>
  );
}