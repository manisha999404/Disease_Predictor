import { useEffect, useRef, useState, useCallback } from "react";

const GREETING = "Hi! I'm Aura, your AI health companion.";

const FOLLOWUP_MESSAGES = [
  "Tell me your symptoms — I'll figure out what's going on.",
  "I've analyzed millions of medical cases. You're in good hands.",
  "I can suggest specialists, remedies, and prevention tips.",
  "Early awareness leads to much better outcomes.",
  "Your health data stays completely private here.",
  "I learn from every conversation to help you better.",
  "Describe how you feel and I'll do the rest.",
];

const TICKLE_MESSAGES = [
  "Hehe... that tickles! 😄",
  "Stop it, stop it! You're making me giggle!",
  "Okay okay, I give up — hahaha!",
  "You found my weak spot! 😂",
  "I'm a ghost, I'm not supposed to laugh!",
  "Hahahaha — please, I can't take it!",
];

function Bubble({ text, position, onDone }) {
  // Start at "idle" so we can kick off the enter animation on next tick
  const [phase, setPhase] = useState("idle");

  useEffect(() => {
    // Kick into "enter" on next frame so the browser registers the initial scale(0)
    const t0 = requestAnimationFrame(() => setPhase("enter"));
    // Hold after spring settles
    const t1 = setTimeout(() => setPhase("hold"), 420);
    // Exit
    const t2 = setTimeout(() => setPhase("exit"), 2800);
    // Unmount
    const t3 = setTimeout(onDone, 3080);
    return () => {
      cancelAnimationFrame(t0);
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  const isRight = position === "right";
  const origin = isRight ? "left center" : "right center";

  const scale =
    phase === "idle"  ? "scale(0)"
    : phase === "enter" ? "scale(1)"
    : phase === "hold"  ? "scale(1)"
    : /* exit */          "scale(0.55)";

  const opacity =
    phase === "idle"  ? 0
    : phase === "exit" ? 0
    : 1;

  const transition =
    phase === "enter"
      ? "transform 0.45s cubic-bezier(0.34,1.65,0.64,1), opacity 0.12s ease"
      : phase === "exit"
      ? "transform 0.22s cubic-bezier(0.55,0,1,0.8), opacity 0.22s ease"
      : "none";

  const wrapStyle = {
    position: "absolute",
    pointerEvents: "none",
    zIndex: 50,
    minWidth: 185,
    maxWidth: 225,
    top: isRight ? "6%" : "30%",
    ...(isRight ? { left: "calc(100% + 18px)" } : { right: "calc(100% + 18px)" }),
    opacity,
    transform: scale,
    transformOrigin: origin,
    transition,
  };

  const tailBase = {
    position: "absolute",
    width: 9,
    height: 9,
    background: "rgba(8, 4, 26, 0.97)",
    border: "1px solid rgba(139, 92, 246, 0.28)",
  };

  const tailStyle = isRight
    ? { ...tailBase, left: -5, top: 13, borderRight: "none", borderTop: "none", transform: "rotate(45deg)" }
    : { ...tailBase, right: -5, top: 13, borderLeft: "none", borderBottom: "none", transform: "rotate(45deg)" };

  return (
    <div style={wrapStyle}>
      <div
        style={{
          background: "rgba(8, 4, 26, 0.97)",
          border: "1px solid rgba(139, 92, 246, 0.28)",
          borderRadius: 11,
          padding: "9px 13px",
          boxShadow: "0 4px 24px rgba(109, 40, 217, 0.25), inset 0 1px 0 rgba(255,255,255,0.04)",
          position: "relative",
        }}
      >
        <p
          style={{
            margin: 0,
            fontSize: 12.5,
            lineHeight: 1.6,
            color: "rgba(226, 232, 240, 0.92)",
            fontFamily: "Inter, system-ui, sans-serif",
            fontWeight: 400,
            letterSpacing: "0.012em",
          }}
        >
          {text}
        </p>
        <div style={tailStyle} />
      </div>
    </div>
  );
}

export function AuraAssistant({
  thinking = false,
  size = 280,
}) {
  const canvasRef = useRef(null);
  const eyeRef = useRef({ x: 0, y: 0 });
  const targetEye = useRef({ x: 0, y: 0 });
  const blinkRef = useRef(false);
  const blinkTimer = useRef(0);
  const nextBlink = useRef(3800 + Math.random() * 1000);
  const tRef = useRef(0);
  const rafRef = useRef(null);
  const offHalo = useRef(null);
  const offFringe = useRef(null);

  const [bubbles, setBubbles] = useState([]);

  const hasGreetedRef = useRef(false);
  const followupIndexRef = useRef(0);
  const rapidClickCountRef = useRef(0);
  const rapidClickTimerRef = useRef(null);
  const tickleIndexRef = useRef(0);

  const removeBubble = useCallback((id) => {
    setBubbles((prev) => prev.filter((b) => b.id !== id));
  }, []);

  const spawnBubble = useCallback((text) => {
    const position = Math.random() < 0.5 ? "right" : "left";
    const id = Date.now() + Math.random();
    setBubbles((prev) => [
      ...prev.filter((b) => b.position !== position),
      { id, text, position },
    ]);
  }, []);

  const handleClick = useCallback(() => {
    rapidClickCountRef.current += 1;
    clearTimeout(rapidClickTimerRef.current);
    rapidClickTimerRef.current = setTimeout(() => {
      rapidClickCountRef.current = 0;
    }, 600);

    if (rapidClickCountRef.current >= 3) {
      const msg = TICKLE_MESSAGES[tickleIndexRef.current % TICKLE_MESSAGES.length];
      tickleIndexRef.current += 1;
      rapidClickCountRef.current = 0;
      spawnBubble(msg);
      return;
    }

    if (!hasGreetedRef.current) {
      hasGreetedRef.current = true;
      spawnBubble(GREETING);
      return;
    }

    const msg = FOLLOWUP_MESSAGES[followupIndexRef.current % FOLLOWUP_MESSAGES.length];
    followupIndexRef.current += 1;
    spawnBubble(msg);
  }, [spawnBubble]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const PAD = size * 0.25;
    const W = size + PAD * 2;
    const H = size + PAD * 2;
    canvas.width = W;
    canvas.height = H;
    canvas.style.position = "absolute";
    canvas.style.top = `-${PAD}px`;
    canvas.style.left = `-${PAD}px`;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    canvas.style.pointerEvents = "none";
    canvas.style.willChange = "transform";

    const MX = W / 2;
    const MY = H / 2 + size * 0.03;

    const makeHaloCanvas = () => {
      const oc = document.createElement("canvas");
      oc.width = W; oc.height = H;
      const ox = oc.getContext("2d");
      ox.filter = "blur(24px)";
      const haloR = size * 0.62;
      const halo = ox.createRadialGradient(MX, MY - 8, 28, MX, MY, haloR);
      halo.addColorStop(0,    "rgba(180,130,255,0.0)");
      halo.addColorStop(0.52, "rgba(100,60,220,0.0)");
      halo.addColorStop(0.72, "rgba(120,80,255,0.24)");
      halo.addColorStop(0.88, "rgba(80,140,255,0.12)");
      halo.addColorStop(1,    "rgba(0,0,0,0)");
      ox.fillStyle = halo;
      ox.beginPath();
      ox.ellipse(MX, MY, haloR, haloR, 0, 0, Math.PI * 2);
      ox.fill();
      return oc;
    };

    const makeCenterGlowCanvas = () => {
      const oc = document.createElement("canvas");
      oc.width = W; oc.height = H;
      const ox = oc.getContext("2d");
      ox.filter = "blur(20px)";
      const centerGlow = ox.createRadialGradient(MX, MY - size * 0.018, 0, MX, MY, size * 0.268);
      centerGlow.addColorStop(0,   "rgba(180,140,255,0.52)");
      centerGlow.addColorStop(0.5, "rgba(120,80,220,0.2)");
      centerGlow.addColorStop(1,   "rgba(0,0,0,0)");
      ox.fillStyle = centerGlow;
      ox.beginPath();
      ox.ellipse(MX, MY - size * 0.018, size * 0.25, size * 0.232, 0, 0, Math.PI * 2);
      ox.fill();
      return oc;
    };

    offHalo.current = makeHaloCanvas();
    offFringe.current = makeCenterGlowCanvas();

    let cachedRect = canvas.getBoundingClientRect();
    const refreshRect = () => { cachedRect = canvas.getBoundingClientRect(); };

    const onMouseMove = (e) => {
      const dx = e.clientX - cachedRect.left - MX;
      const dy = e.clientY - cachedRect.top - MY;
      const d = Math.hypot(dx, dy) || 1;
      const k = Math.min(6, d / 18) / d;
      targetEye.current = { x: dx * k, y: dy * k };
    };
    const onMouseLeave = () => { targetEye.current = { x: 0, y: 0 }; };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("scroll", refreshRect, { passive: true });
    window.addEventListener("resize", refreshRect, { passive: true });
    document.addEventListener("mouseleave", onMouseLeave);

    function blobR(angle, time) {
      const base = size * 0.335;
      const w1 = Math.sin(angle * 2 + time * 0.55) * (size * 0.028);
      const w2 = Math.sin(angle * 3 - time * 0.38) * (size * 0.016);
      const w3 = Math.sin(angle * 4 + time * 0.72) * (size * 0.009);
      const breathe = Math.sin(time * 0.28) * (size * 0.011);
      const squish = Math.cos(angle * 1 + time * 0.18) * (size * 0.013);
      return base + w1 + w2 + w3 + breathe + squish;
    }

    function getPoints(time, n = 80, scale = 1.0) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2;
        const r = blobR(a, time) * scale;
        pts.push({ x: MX + Math.cos(a) * r, y: MY + Math.sin(a) * r });
      }
      return pts;
    }

    function tracePath(pts) {
      const n = pts.length;
      ctx.beginPath();
      ctx.moveTo((pts[0].x + pts[n - 1].x) / 2, (pts[0].y + pts[n - 1].y) / 2);
      for (let i = 0; i < n; i++) {
        const p = pts[i];
        const q = pts[(i + 1) % n];
        ctx.quadraticCurveTo(p.x, p.y, (p.x + q.x) / 2, (p.y + q.y) / 2);
      }
      ctx.closePath();
    }

    let frameCount = 0;

function drawFrame() {
  rafRef.current = requestAnimationFrame(drawFrame);
  frameCount++;
  if (frameCount % 2 !== 0) return; // ~30fps
      ctx.clearRect(0, 0, W, H);

      const t = tRef.current;
      const pts = getPoints(t);
      const outerPts = getPoints(t, 120, 1.18);

      ctx.drawImage(offHalo.current, 0, 0);

      ctx.save();
      ctx.filter = "blur(12px)";
      const fringeGrad = ctx.createRadialGradient(MX, MY, size * 0.24, MX, MY, size * 0.46);
      fringeGrad.addColorStop(0,    "rgba(0,0,0,0)");
      fringeGrad.addColorStop(0.55, "rgba(148,100,255,0.18)");
      fringeGrad.addColorStop(0.8,  "rgba(100,60,230,0.28)");
      fringeGrad.addColorStop(1,    "rgba(0,0,0,0)");
      ctx.fillStyle = fringeGrad;
      tracePath(outerPts);
      ctx.fill();
      ctx.filter = "none";
      ctx.restore();

      ctx.save();
      ctx.filter = "blur(8px)";
      const edgePts = getPoints(t, 120, 1.06);
      const edgeGlow = ctx.createRadialGradient(MX, MY, size * 0.18, MX, MY, size * 0.42);
      edgeGlow.addColorStop(0,   "rgba(0,0,0,0)");
      edgeGlow.addColorStop(0.7, "rgba(110,60,210,0.48)");
      edgeGlow.addColorStop(1,   "rgba(0,0,0,0)");
      ctx.fillStyle = edgeGlow;
      tracePath(edgePts);
      ctx.fill();
      ctx.filter = "none";
      ctx.restore();

      const bodyGrad = ctx.createRadialGradient(
        MX - size * 0.065, MY - size * 0.079, size * 0.03,
        MX + size * 0.018, MY + size * 0.036, size * 0.386
      );
      bodyGrad.addColorStop(0,    "#c8a8ff");
      bodyGrad.addColorStop(0.18, "#a070f0");
      bodyGrad.addColorStop(0.45, "#7740d8");
      bodyGrad.addColorStop(0.72, "#5520b0");
      bodyGrad.addColorStop(1,    "#2d0870");
      ctx.fillStyle = bodyGrad;
      tracePath(pts);
      ctx.fill();

      ctx.save();
      ctx.globalCompositeOperation = "multiply";
      ctx.filter = "blur(5px)";
      const shadowGrad = ctx.createRadialGradient(
        MX + size * 0.036, MY + size * 0.107, size * 0.036,
        MX, MY + size * 0.089, size * 0.304
      );
      shadowGrad.addColorStop(0,   "rgba(10,0,40,0)");
      shadowGrad.addColorStop(0.5, "rgba(10,0,40,0.35)");
      shadowGrad.addColorStop(1,   "rgba(0,0,0,0.5)");
      ctx.fillStyle = shadowGrad;
      tracePath(pts);
      ctx.fill();
      ctx.filter = "none";
      ctx.globalCompositeOperation = "source-over";
      ctx.restore();

      ctx.save();
      ctx.filter = "blur(16px)";
      const hiGrad = ctx.createRadialGradient(
        MX - size * 0.089, MY - size * 0.125, 0,
        MX - size * 0.064, MY - size * 0.089, size * 0.186
      );
      hiGrad.addColorStop(0,   "rgba(255,255,255,0.75)");
      hiGrad.addColorStop(0.4, "rgba(210,190,255,0.38)");
      hiGrad.addColorStop(1,   "rgba(0,0,0,0)");
      ctx.fillStyle = hiGrad;
      ctx.beginPath();
      ctx.ellipse(MX - size * 0.079, MY - size * 0.107, size * 0.164, size * 0.136, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.filter = "none";
      ctx.restore();

      ctx.drawImage(offFringe.current, 0, 0);

      ctx.save();
      tracePath(pts);
      ctx.clip();

      eyeRef.current.x += (targetEye.current.x - eyeRef.current.x) * 0.12;
      eyeRef.current.y += (targetEye.current.y - eyeRef.current.y) * 0.12;

      const eyeOffX = size * 0.088;
      const eyeOffY = size * 0.038;
      const eyeH    = blinkRef.current ? 2 : size * 0.088;
      const eyeW    = size * 0.044;

      const eyes = [
        { bx: MX - eyeOffX + eyeRef.current.x, by: MY - eyeOffY + eyeRef.current.y, tilt: -6 },
        { bx: MX + eyeOffX + eyeRef.current.x, by: MY - eyeOffY + eyeRef.current.y, tilt:  6 },
      ];

      for (const eye of eyes) {
        ctx.save();
        ctx.filter = "blur(10px)";
        const eg = ctx.createRadialGradient(eye.bx, eye.by, 0, eye.bx, eye.by, size * 0.10);
        eg.addColorStop(0,   "rgba(255,255,255,0.9)");
        eg.addColorStop(0.5, "rgba(220,210,255,0.5)");
        eg.addColorStop(1,   "rgba(0,0,0,0)");
        ctx.fillStyle = eg;
        ctx.beginPath();
        ctx.ellipse(eye.bx, eye.by, size * 0.10, size * 0.12, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.filter = "none";
        ctx.restore();

        ctx.save();
        ctx.translate(eye.bx, eye.by);
        ctx.rotate((eye.tilt * Math.PI) / 180);
        const yt = blinkRef.current ? -1 : -eyeH / 2;

        ctx.save();
        ctx.filter = "blur(3px)";
        ctx.fillStyle = "rgba(220,220,255,0.88)";
        ctx.beginPath();
        ctx.roundRect(-eyeW / 2 - 2, yt - 2, eyeW + 4, eyeH + 4, (eyeW + 4) / 2);
        ctx.fill();
        ctx.filter = "none";
        ctx.restore();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.roundRect(-eyeW / 2, yt, eyeW, eyeH, eyeW / 2);
        ctx.fill();
        ctx.restore();
      }

      ctx.restore();

      blinkTimer.current += 16;
      if (blinkTimer.current > nextBlink.current) {
        blinkRef.current = true;
        blinkTimer.current = 0;
        nextBlink.current = (thinking ? 1400 : 3400) + Math.random() * 1200;
        setTimeout(() => { blinkRef.current = false; }, 140);
      }

      tRef.current += thinking ? 0.032 : 0.024;
    }

    rafRef.current = requestAnimationFrame(drawFrame);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("scroll", refreshRect);
      window.removeEventListener("resize", refreshRect);
      document.removeEventListener("mouseleave", onMouseLeave);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [size, thinking]);

  return (
    <div
      onClick={handleClick}
      style={{
        position: "relative",
        width: size,
        height: size,
        background: "transparent",
        overflow: "visible",
        flexShrink: 0,
        cursor: "pointer",
      }}
    >
      <canvas ref={canvasRef} className="aura-canvas" />

      {bubbles.map((b) => (
        <Bubble
          key={b.id}
          text={b.text}
          position={b.position}
          onDone={() => removeBubble(b.id)}
        />
      ))}
    </div>
  );
}