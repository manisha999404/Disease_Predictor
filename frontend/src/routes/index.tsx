import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Sparkles, X, Plus, ArrowRight, ChevronDown, Heart, Shield, Leaf,
  Stethoscope, MapPin, Activity, Brain, Check, AlertTriangle,
} from "lucide-react";
import { AuraAssistant } from "@/components/AuraAssistant";
import { BackgroundEffects } from "@/components/BackgroundEffects";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AURA — Your Intelligent AI Health Companion" },
      { name: "description", content: "Describe symptoms and receive AI-powered disease prediction, prevention, remedies, specialist recommendations, and risk assessment." },
      { property: "og:title", content: "AURA — AI Health Companion" },
      { property: "og:description", content: "AI-powered disease prediction & personalized health guidance." },
    ],
  }),
  component: Index,
});

type Prediction = {
  disease: string;
  probability: number;
  risk: "Mild" | "Moderate" | "Severe";
  icon: string;
  prevention: string[];
  remedies: { emoji: string; name: string; note: string }[];
  specialist: string;
  explanation: string[];
};

const SUGGESTIONS = ["fever", "cough", "headache", "sore throat", "body pain", "fatigue", "nausea", "chills"];
const ACCENT = ["#8B5CF6", "#60A5FA", "#22D3EE"] as const;

function mockPredict(symptoms: string[]): Prediction[] {
  const s = symptoms.join(" ").toLowerCase();
  const base: Prediction[] = [
    {
      disease: "Influenza",
      probability: 78,
      risk: "Moderate",
      icon: "🤒",
      prevention: ["Stay hydrated", "Wash hands frequently", "Get adequate sleep", "Avoid crowded places"],
      remedies: [
        { emoji: "🍵", name: "Ginger Tea", note: "Soothes the throat" },
        { emoji: "🍯", name: "Honey & Warm Water", note: "Eases coughing" },
        { emoji: "💧", name: "Warm Fluids", note: "Maintains hydration" },
      ],
      specialist: "General Physician",
      explanation: [
        "Based on the symptoms you entered, Influenza appears to be the strongest match.",
        "This prediction is informational and should not replace professional medical advice.",
        "Please review the prevention guidance below for a safer recovery.",
      ],
    },
    {
      disease: "Common Cold", probability: 64, risk: "Mild", icon: "🤧",
      prevention: ["Rest well", "Drink warm fluids", "Use saline rinse"],
      remedies: [
        { emoji: "🧄", name: "Garlic Broth", note: "Natural antimicrobial" },
        { emoji: "🍋", name: "Lemon & Honey", note: "Vitamin C boost" },
      ],
      specialist: "General Physician",
      explanation: [
        "Common Cold matches your symptom profile at 64%.",
        "This is informational and should not replace professional advice.",
        "Rest and hydration are your best allies.",
      ],
    },
    {
      disease: "Viral Pharyngitis", probability: 41, risk: "Mild", icon: "🦠",
      prevention: ["Gargle salt water", "Avoid cold beverages"],
      remedies: [
        { emoji: "🧂", name: "Salt Water Gargle", note: "Reduces inflammation" },
        { emoji: "🍵", name: "Chamomile Tea", note: "Soothes the throat" },
      ],
      specialist: "ENT Specialist",
      explanation: [
        "Viral Pharyngitis is a possible match at 41%.",
        "This is informational only — consult a doctor if symptoms worsen.",
        "Throat care and rest are key.",
      ],
    },
  ];
  if (s.includes("chest") || s.includes("breath")) {
    base[0] = { ...base[0], disease: "Bronchitis", probability: 71, risk: "Severe", specialist: "Pulmonologist" };
  }
  return base;
}

function Index() {
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<Prediction[] | null>(null);
  const [loadingMsg, setLoadingMsg] = useState(0);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const resultsRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLDivElement>(null);
  const textInputRef = useRef<HTMLDivElement>(null);

  const messages = [
    "Analyzing symptoms…",
    "Matching disease patterns…",
    "Reviewing medical indicators…",
    "Consulting AI health knowledge…",
    "Generating recommendations…",
  ];

  useEffect(() => {
    if (!analyzing) return;
    const t = setInterval(() => setLoadingMsg((i) => (i + 1) % messages.length), 900);
    return () => clearInterval(t);
  }, [analyzing]);

  const addSymptom = (s?: string) => {
    const v = (s ?? textInputRef.current?.textContent ?? "").trim().toLowerCase();
    if (!v || symptoms.includes(v)) return;
    setSymptoms((prev) => [...prev, v]);
    if (textInputRef.current) textInputRef.current.textContent = "";
  };

  const removeSymptom = (s: string) => {
    setSymptoms((prev) => prev.filter((x) => x !== s));
  };

  const analyze = async () => {
    if (symptoms.length === 0) return;
    setAnalyzing(true);
    setResults(null);
    setSelectedIdx(0);
    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: symptoms.join(", ") }),
      });
      if (!res.ok) throw new Error(`Backend error ${res.status}`);
      const raw: Array<{
        disease: string;
        probability: string;
        prevention?: string[];
        remedies?: string[];
        specialist?: string;
        risk?: "Mild" | "Moderate" | "Severe";
      }> = await res.json();

      const DISEASE_ICONS: Record<string, string> = {
        influenza: "🤒",
        "common cold": "🤧",
        bronchitis: "🫁",
      };

      const mapped: Prediction[] = raw.map((p) => ({
        disease: p.disease,
        probability: Math.round(parseFloat(p.probability.replace("%", "")) || 0),
        risk: p.risk ?? "Moderate",
        icon: DISEASE_ICONS[p.disease.toLowerCase()] ?? "🦠",
        prevention: p.prevention ?? [],
        remedies: (p.remedies ?? []).map((r) => ({ emoji: "🩺", name: r, note: "" })),
        specialist: p.specialist ?? "General Physician",
        explanation: [
          `Based on the symptoms you entered, ${p.disease} appears to be a strong match.`,
          "This prediction is informational and should not replace professional medical advice.",
          "Please review the prevention guidance below for a safer recovery.",
        ],
      }));

      setResults(mapped);
    } catch (err) {
      console.error("Prediction failed:", err);
      setResults(mockPredict(symptoms));
    } finally {
      setAnalyzing(false);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
    }
  };

  const selected = results?.[selectedIdx] ?? results?.[0];
  const riskColor = (r: string) =>
    r === "Severe" ? "var(--danger)" : r === "Moderate" ? "var(--warning)" : "var(--success)";

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <BackgroundEffects />

      {/* Navbar */}
      <nav className="fixed top-0 inset-x-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between glass rounded-2xl px-5 py-3">
          <a href="#" className="flex items-center gap-2 font-display text-lg font-bold">
            <span className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg,#8B5CF6,#22D3EE)" }}>
              <Sparkles size={18} className="text-white" />
            </span>
            <span className="text-gradient">AURA</span>
          </a>
          <div className="hidden md:flex gap-8 text-sm text-muted-foreground">
            <a href="#assistant" className="hover:text-white transition">Assistant</a>
            <a href="#check" className="hover:text-white transition">Health Check</a>
            <a href="#how" className="hover:text-white transition">How it works</a>
            <a href="#disclaimer" className="hover:text-white transition">Safety</a>
          </div>
          <button
            onClick={() => sectionRef.current?.scrollIntoView({ behavior: "smooth" })}
            className="text-sm font-medium px-4 py-2 rounded-xl text-white transition hover:scale-105"
            style={{ background: "linear-gradient(135deg,#8B5CF6,#60A5FA)" }}
          >
            Start
          </button>
        </div>
      </nav>

      {/* HERO */}
      <section id="assistant" className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}
          className="flex items-center gap-2 glass rounded-full px-4 py-1.5 text-xs text-muted-foreground mb-8"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          AI Health Companion · Online
        </motion.div>

        <AuraAssistant thinking={analyzing} size={300} />

        <motion.h1
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.8 }}
          className="mt-10 text-6xl md:text-8xl font-bold text-gradient text-center"
        >
          Meet Aura
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.8 }}
          className="mt-4 text-lg md:text-2xl text-white/80 font-display text-center max-w-2xl"
        >
          Your Intelligent AI Health Companion
        </motion.p>
        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6, duration: 0.8 }}
          className="mt-6 text-base text-muted-foreground max-w-2xl text-center leading-relaxed"
        >
          Describe your symptoms and receive AI-powered disease prediction, personalized prevention guidance,
          safe home remedies, specialist recommendations, and risk assessment.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8, duration: 0.8 }}
          className="mt-10 flex flex-wrap gap-4 justify-center"
        >
          <button
            onClick={() => sectionRef.current?.scrollIntoView({ behavior: "smooth" })}
            className="group relative px-7 py-3.5 rounded-2xl font-medium text-white overflow-hidden glow-primary transition hover:scale-105"
            style={{ background: "linear-gradient(135deg,#8B5CF6,#60A5FA)" }}
          >
            <span className="relative z-10 flex items-center gap-2">Start Health Check <ArrowRight size={18} /></span>
          </button>
          <a href="#how" className="px-7 py-3.5 rounded-2xl font-medium text-white glass hover:bg-white/5 transition">
            Learn More
          </a>
        </motion.div>

        <motion.a
          href="#check"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}
          className="absolute bottom-8 flex flex-col items-center gap-2 text-muted-foreground text-xs"
        >
          Scroll
          <ChevronDown className="animate-bounce" size={16} />
        </motion.a>
      </section>

      {/* SYMPTOM INPUT */}
      <section id="check" ref={sectionRef as any} className="relative py-32 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <div className="inline-flex items-center gap-2 glass rounded-full px-3 py-1 text-xs text-accent mb-4">
              <Brain size={12} /> AI ANALYSIS
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gradient">Describe Your Symptoms</h2>
            <p className="mt-3 text-muted-foreground">Enter one or multiple symptoms to begin AI analysis.</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.8 }}
            className="glass rounded-3xl p-8 relative"
            style={{ boxShadow: "0 30px 80px -20px rgba(139,92,246,0.4)" }}
          >
            <div className="flex flex-wrap gap-2 min-h-[3rem] mb-4">
              <AnimatePresence>
                {symptoms.map((s) => (
                  <motion.span
                    key={s}
                    initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm text-white"
                    style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.25), rgba(96,165,250,0.2))", border: "1px solid rgba(139,92,246,0.4)", boxShadow: "0 0 16px rgba(139,92,246,0.3)" }}
                  >
                    {s}
                    <button onClick={() => removeSymptom(s)} className="hover:scale-110 transition">
                      <X size={14} />
                    </button>
                  </motion.span>
                ))}
                {symptoms.length === 0 && (
                  <span className="text-sm text-muted-foreground italic">No symptoms yet — try the suggestions below.</span>
                )}
              </AnimatePresence>
            </div>

            <div className="flex gap-2">
              <div
                ref={textInputRef}
                contentEditable
                suppressContentEditableWarning
                data-placeholder="Type a symptom and press Enter…"
                onKeyDown={(e) => {
                  if (e.key === "Enter") { e.preventDefault(); addSymptom(); }
                  if (e.key === "Backspace" && !e.currentTarget.textContent && symptoms.length) {
                    removeSymptom(symptoms[symptoms.length - 1]);
                  }
                }}
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/30 transition empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground empty:before:pointer-events-none"
              />
              <button
                onClick={() => addSymptom()}
                className="px-4 rounded-xl glass hover:bg-white/10 transition flex items-center gap-1 text-sm"
              >
                <Plus size={16} /> Add
              </button>
            </div>

            <div className="flex flex-wrap gap-2 mt-4">
              {SUGGESTIONS.filter((s) => !symptoms.includes(s)).map((s) => (
                <button
                  key={s}
                  onClick={() => addSymptom(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-white/10 text-muted-foreground hover:text-white hover:border-primary/50 hover:bg-primary/10 transition"
                >
                  + {s}
                </button>
              ))}
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              onClick={analyze}
              disabled={analyzing || symptoms.length === 0}
              className="mt-8 w-full py-4 rounded-2xl font-semibold text-white text-lg relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed transition"
              style={{ background: "linear-gradient(135deg,#8B5CF6,#60A5FA,#22D3EE)", backgroundSize: "200% 200%", boxShadow: "0 10px 40px -10px rgba(139,92,246,0.6), 0 0 30px rgba(96,165,250,0.3)" }}
            >
              <span className="relative z-10 flex items-center justify-center gap-2">
                {analyzing ? <><Activity className="animate-spin" size={18} /> Analyzing…</> : <><Sparkles size={18} /> Analyze Symptoms</>}
              </span>
            </motion.button>
          </motion.div>

          <AnimatePresence>
            {analyzing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="mt-12 flex flex-col items-center gap-6"
              >
                <AuraAssistant thinking size={180} />
                <AnimatePresence mode="wait">
                  <motion.p
                    key={loadingMsg}
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    className="text-lg font-display text-white/90"
                  >
                    {messages[loadingMsg]}
                  </motion.p>
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* RESULTS */}
      <AnimatePresence>
        {results && selected && (
          <motion.div
            ref={resultsRef}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="relative px-6 pb-32"
          >
            {/* ── Disease selector cards ── */}
            <section className="max-w-6xl mx-auto">
              <SectionHeader kicker="PREDICTIONS" title="Top Disease Matches" />
              <div className="grid md:grid-cols-3 gap-5">
                {results.map((r, i) => {
                  const accent = ACCENT[i] ?? "#8B5CF6";
                  const isSelected = i === selectedIdx;
                  return (
                    <motion.button
                      key={r.disease}
                      onClick={() => setSelectedIdx(i)}
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.12, duration: 0.6 }}
                      className="text-left w-full glass rounded-2xl p-6 transition-all duration-300 relative overflow-hidden group focus:outline-none"
                      style={{
                        border: isSelected ? `1.5px solid ${accent}99` : "1.5px solid rgba(255,255,255,0.06)",
                        boxShadow: isSelected ? `0 0 36px -8px ${accent}77` : "none",
                        transform: isSelected ? "translateY(-6px)" : "translateY(0)",
                      }}
                    >
                      {/* Glow orb */}
                      <div
                        className="absolute -top-10 -right-10 w-40 h-40 rounded-full blur-3xl pointer-events-none transition-opacity duration-300"
                        style={{ background: accent, opacity: isSelected ? 0.4 : 0.15 }}
                      />

                      <div className="flex items-start justify-between relative">
                        <div className="text-4xl">{r.icon}</div>
                        <div className="flex flex-col items-end gap-1.5">
                          <span
                            className="text-xs px-2.5 py-1 rounded-full"
                            style={{ background: `${riskColor(r.risk)}22`, color: riskColor(r.risk), border: `1px solid ${riskColor(r.risk)}55` }}
                          >
                            {r.risk} risk
                          </span>
                          {isSelected && (
                            <motion.span
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              className="text-xs px-2 py-0.5 rounded-full font-medium text-white"
                              style={{ background: `${accent}44`, border: `1px solid ${accent}88` }}
                            >
                              Viewing ✦
                            </motion.span>
                          )}
                        </div>
                      </div>

                      <h3 className="mt-4 text-xl font-bold">{r.disease}</h3>

                      <div className="mt-4">
                        <div className="flex justify-between text-xs text-muted-foreground mb-2">
                          <span>Match probability</span>
                          <span className="font-semibold text-white">{r.probability}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${r.probability}%` }}
                            transition={{ delay: 0.3 + i * 0.12, duration: 1, ease: "easeOut" }}
                            className="h-full rounded-full"
                            style={{
                              background: `linear-gradient(90deg, ${accent}, #22D3EE)`,
                              boxShadow: `0 0 10px ${accent}99`,
                            }}
                          />
                        </div>
                      </div>

                      {!isSelected && (
                        <p className="mt-3 text-xs text-muted-foreground group-hover:text-white/60 transition">
                          Tap to view details →
                        </p>
                      )}
                    </motion.button>
                  );
                })}
              </div>

              <motion.p
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
                className="text-center text-base text-muted-foreground mt-7"
              >
                Select a disease card to explore its prevention, remedies, and specialist.
              </motion.p>
            </section>

            {/* ── Details — animate when switching diseases ── */}
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedIdx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.4 }}
              >
                {/* Aura says */}
                <section className="max-w-5xl mx-auto mt-24">
                  <SectionHeader kicker="AURA SAYS" title="What this means" />
                  <div className="flex flex-col md:flex-row gap-8 items-center md:items-start">
                    <div className="shrink-0"><AuraAssistant size={180} /></div>
                    <div className="flex-1 space-y-3">
                      {selected.explanation.map((line, i) => (
                        <ChatBubble key={`${selectedIdx}-${i}`} text={line} delay={i * 0.8} />
                      ))}
                    </div>
                  </div>
                </section>

                {/* Prevention */}
                <section className="max-w-6xl mx-auto mt-24">
                  <SectionHeader kicker="PREVENTION" title="Personalized guidance" icon={<Shield size={12} />} />
                  {selected.prevention.length === 0 ? (
                    <p className="text-center text-muted-foreground text-sm">No prevention data available for this disease.</p>
                  ) : (
                    <div className="grid sm:grid-cols-2 gap-4">
                      {selected.prevention.map((p, i) => (
                        <motion.div
                          key={`${selectedIdx}-prev-${i}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.07 }}
                          className="glass rounded-xl p-5 flex items-center gap-4 hover:bg-white/5 transition"
                        >
                          <div
                            className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                            style={{ background: "linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,211,238,0.2))", border: "1px solid rgba(34,197,94,0.3)" }}
                          >
                            <Check className="text-emerald-400" size={18} />
                          </div>
                          <span className="text-white/90">{p}</span>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </section>

                {/* Remedies */}
                <section className="max-w-6xl mx-auto mt-24">
                  <SectionHeader kicker="HOME REMEDIES" title="Safe and natural" icon={<Leaf size={12} />} />
                  {selected.remedies.length === 0 ? (
                    <p className="text-center text-muted-foreground text-sm">No remedy data available for this disease.</p>
                  ) : (
                    <div className="grid md:grid-cols-3 gap-5">
                      {selected.remedies.map((r, i) => (
                        <motion.div
                          key={`${selectedIdx}-rem-${i}`}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.09 }}
                          className="glass rounded-2xl p-6 text-center hover:-translate-y-1 transition relative"
                          style={{ boxShadow: "0 0 30px -15px rgba(245,158,11,0.4)" }}
                        >
                          <div className="text-5xl mb-3">{r.emoji}</div>
                          <h4 className="font-semibold text-lg">{r.name}</h4>
                          <p className="text-sm text-muted-foreground mt-1">{r.note}</p>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </section>

                {/* Specialist + Risk */}
                <section className="max-w-6xl mx-auto mt-24 grid lg:grid-cols-2 gap-6">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass rounded-3xl p-8 relative overflow-hidden"
                  >
                    <div className="text-xs text-accent flex items-center gap-1 mb-2">
                      <Stethoscope size={12} /> RECOMMENDED SPECIALIST
                    </div>
                    <div className="flex items-center gap-5 mt-4">
                      <div
                        className="w-20 h-20 rounded-2xl flex items-center justify-center shrink-0"
                        style={{ background: `linear-gradient(135deg, ${ACCENT[selectedIdx] ?? "#8B5CF6"}, #22D3EE)`, boxShadow: `0 0 30px ${ACCENT[selectedIdx] ?? "#8B5CF6"}66` }}
                      >
                        <Stethoscope className="text-white" size={36} />
                      </div>
                      <div>
                        <h3 className="text-2xl font-bold">{selected.specialist}</h3>
                        <p className="text-sm text-muted-foreground">Based on your symptom profile</p>
                      </div>
                    </div>
                    <a
                      href={`https://www.google.com/maps/search/${encodeURIComponent(selected.specialist)}+near+me`}
                      target="_blank" rel="noreferrer"
                      className="mt-6 inline-flex items-center gap-2 px-5 py-3 rounded-xl text-white font-medium transition hover:scale-105"
                      style={{ background: `linear-gradient(135deg, ${ACCENT[selectedIdx] ?? "#8B5CF6"}, #60A5FA)`, boxShadow: `0 0 20px ${ACCENT[selectedIdx] ?? "#8B5CF6"}55` }}
                    >
                      <MapPin size={16} /> Find Nearby Doctor
                    </a>
                  </motion.div>
                  <RiskMeter risk={selected.risk} />
                </section>
              </motion.div>
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>

      {/* How it works */}
      <section id="how" className="relative py-32 px-6">
        <div className="max-w-6xl mx-auto">
          <SectionHeader kicker="HOW IT WORKS" title="From symptoms to clarity" />
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { icon: <Heart size={20} />, title: "Describe", desc: "Add the symptoms you're experiencing in natural language." },
              { icon: <Brain size={20} />, title: "Analyze", desc: "Aura matches patterns against a curated medical knowledge base." },
              { icon: <Shield size={20} />, title: "Act", desc: "Get prevention, remedies, risk and a specialist suggestion." },
            ].map((s, i) => (
              <motion.div
                key={s.title}
                initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                className="glass rounded-2xl p-6"
              >
                <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 text-white"
                  style={{ background: "linear-gradient(135deg,#8B5CF6,#60A5FA)" }}>{s.icon}</div>
                <h3 className="font-bold text-lg">{s.title}</h3>
                <p className="text-sm text-muted-foreground mt-2">{s.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer id="disclaimer" className="relative px-6 pb-12">
        <div className="max-w-5xl mx-auto glass rounded-3xl p-8 text-center">
          <div className="inline-flex items-center gap-2 text-xs text-warning mb-3">
            <AlertTriangle size={14} /> MEDICAL DISCLAIMER
          </div>
          <p className="text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            This tool provides AI-assisted informational guidance and should not be considered a substitute for
            professional medical diagnosis, treatment, or advice. Always consult a qualified healthcare provider.
          </p>
        </div>
        <p className="text-center text-xs text-muted-foreground mt-8">© 2026 AURA · Crafted with care</p>
      </footer>
    </div>
  );
}

function SectionHeader({ kicker, title, icon }: { kicker: string; title: string; icon?: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
      className="text-center mb-10"
    >
      <div className="inline-flex items-center gap-1.5 glass rounded-full px-3 py-1 text-xs text-accent mb-3">
        {icon} {kicker}
      </div>
      <h2 className="text-3xl md:text-5xl font-bold text-gradient">{title}</h2>
    </motion.div>
  );
}

function ChatBubble({ text, delay = 0 }: { text: string; delay?: number }) {
  const chars = text.length;
  const duration = (chars * 0.022).toFixed(2);
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay }}
      className="glass rounded-2xl rounded-tl-sm px-5 py-4 inline-block max-w-2xl overflow-hidden"
    >
      <p
        className="text-white/90 leading-relaxed whitespace-pre-wrap overflow-hidden"
        style={{
          width: `${chars}ch`,
          maxWidth: "100%",
          borderRight: "2px solid rgba(255,255,255,0.4)",
          animation: `aura-typing ${duration}s steps(${chars},end) ${delay}s both, aura-caret 0.7s step-end ${parseFloat(duration) + delay}s 4`,
        }}
      >
        {text}
      </p>
    </motion.div>
  );
}

function RiskMeter({ risk }: { risk: "Mild" | "Moderate" | "Severe" }) {
  const value = risk === "Mild" ? 25 : risk === "Moderate" ? 60 : 90;
  const color = risk === "Severe" ? "#EF4444" : risk === "Moderate" ? "#F59E0B" : "#22C55E";
  const desc = risk === "Mild" ? "Self-care may be sufficient." :
    risk === "Moderate" ? "Monitor symptoms and consult a physician if they persist." :
    "Seek medical attention promptly.";
  const angle = useMemo(() => -90 + (value / 100) * 180, [value]);
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-3xl p-8 relative overflow-hidden"
    >
      <div className="text-xs flex items-center gap-1 mb-2" style={{ color }}><Activity size={12} /> RISK ASSESSMENT</div>
      <div className="flex flex-col items-center mt-2">
        <svg viewBox="0 0 200 120" className="w-full max-w-xs">
          <defs>
            <linearGradient id="gauge" x1="0" x2="1">
              <stop offset="0%" stopColor="#22C55E" />
              <stop offset="50%" stopColor="#F59E0B" />
              <stop offset="100%" stopColor="#EF4444" />
            </linearGradient>
          </defs>
          <path d="M20 110 A 80 80 0 0 1 180 110" stroke="rgba(255,255,255,0.08)" strokeWidth="14" fill="none" strokeLinecap="round" />
          <path d="M20 110 A 80 80 0 0 1 180 110" stroke="url(#gauge)" strokeWidth="14" fill="none" strokeLinecap="round"
            strokeDasharray="251" strokeDashoffset={251 - (value / 100) * 251} style={{ transition: "stroke-dashoffset 1.2s ease-out" }} />
          <motion.line
            x1="100" y1="110" x2="100" y2="40"
            stroke={color} strokeWidth="3" strokeLinecap="round"
            style={{ transformOrigin: "100px 110px", filter: `drop-shadow(0 0 8px ${color})` }}
            initial={{ rotate: -90 }} animate={{ rotate: angle }} transition={{ duration: 1.2, ease: "easeOut" }}
          />
          <circle cx="100" cy="110" r="6" fill={color} />
        </svg>
        <div className="text-3xl font-bold mt-2" style={{ color }}>{risk}</div>
        <p className="text-sm text-muted-foreground text-center mt-2 max-w-xs">{desc}</p>
      </div>
    </motion.div>
  );
}