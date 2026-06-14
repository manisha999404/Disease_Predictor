import { z } from "zod";

export const RawPrediction = z.object({
  disease: z.string(),
  probability: z.string(),
  prevention: z.array(z.string()).optional().default([]),
  remedies: z.array(z.string()).optional().default([]),
  specialist: z.string().optional().default("General Physician"),
  risk: z.enum(["Mild", "Moderate", "Severe"]).optional().default("Moderate"),
});

export const PredictInput = z.object({
  symptoms: z.array(z.string().min(1)).min(1),
});

export const DISEASE_ICONS: Record<string, string> = {
  influenza: "🤒",
  "common cold": "🤧",
  bronchitis: "🫁",
  default: "🦠",
};