export const MODEL_ALIASES = {
  "opus-4": "claude-opus-4",
  "opus-4.6": "claude-opus-4-6",
  "opus-4-6": "claude-opus-4-6",
  "sonnet-4": "claude-sonnet-4",
  "sonnet-4.6": "claude-sonnet-4-6",
  "sonnet-4-6": "claude-sonnet-4-6",
  haiku: "claude-3-5-haiku-latest",
  "haiku-3.5": "claude-3-5-haiku-latest",
};

function resolveAlias(model) {
  if (!model) {
    return model;
  }
  const key = String(model).trim().toLowerCase();
  return MODEL_ALIASES[key] ?? model;
}

export function normalizeAnthropicModel(model) {
  if (!model) {
    return model;
  }
  let normalized = resolveAlias(String(model).trim());
  normalized = normalized
    .replace(/^anthropic\//i, "")
    .replace(/^claude\//i, "claude-")
    .trim();
  return normalized;
}

export function isAnthropicModel(model) {
  if (!model) {
    return false;
  }
  const candidate = normalizeAnthropicModel(model).toLowerCase();
  return /claude|opus|sonnet|haiku/.test(candidate);
}

export function detectModelProvider(model) {
  if (!model) {
    return null;
  }
  const lower = String(model).trim().toLowerCase();
  if (isAnthropicModel(lower) || lower.startsWith("anthropic/")) {
    return "anthropic";
  }
  if (
    lower.startsWith("openai/") ||
    /^gpt[-/]/.test(lower) ||
    /^o[1-9](-|$)/.test(lower) ||
    lower.includes("gpt")
  ) {
    return "openai";
  }
  return null;
}
