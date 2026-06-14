import { z } from "zod";
import { RawPrediction, PredictInput, DISEASE_ICONS } from "./health.schemas";

// Plain async function — no createServerFn, no RPC interceptors, no global
// event listeners. createServerFn from @tanstack/react-start was registering
// a global fetch/interaction interceptor that fired on every input focus = freeze.
export const predictDiseases = async (input: z.infer<typeof PredictInput>) => {
  const data = PredictInput.parse(input);
  
  const backendUrl = import.meta.env.VITE_FLASK_API_URL ?? "http://127.0.0.1:5000";

  const res = await fetch(`${backendUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms: data.symptoms.join(", ") }),
  });

  if (!res.ok) {
    throw new Error(`Backend prediction failed (${res.status})`);
  }

  const raw = await res.json();
  const parsed = z.array(RawPrediction).parse(raw);

  return parsed.map((p) => {
    const probNum = parseFloat(p.probability.replace("%", "")) || 0;
    const icon = DISEASE_ICONS[p.disease.toLowerCase()] ?? DISEASE_ICONS.default;

    return {
      disease: p.disease,
      probability: Math.round(probNum),
      risk: p.risk,
      icon,
      prevention: p.prevention,
      remedies: p.remedies.map((r) => ({ emoji: "🩺", name: r, note: "" })),
      specialist: p.specialist,
      explanation: [
        `Based on the symptoms you entered, ${p.disease} appears to be a strong match.`,
        "This prediction is informational and should not replace professional medical advice.",
        "Please review the prevention guidance below for a safer recovery.",
      ],
    };
  });
};