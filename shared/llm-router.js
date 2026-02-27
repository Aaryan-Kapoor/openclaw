import { runAnthropicAgentPrompt } from "./anthropic-agent-sdk.js";
import { estimateTokensFromChars, logLlmCall } from "./interaction-store.js";
import { detectModelProvider, isAnthropicModel } from "./model-utils.js";

async function runOpenAiPrompt({ model, prompt, timeoutMs = 60_000 }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set for non-Anthropic model routing.");
  }

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(new Error("OpenAI request timed out")), timeoutMs);

  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model || "gpt-4o-mini",
        messages: [{ role: "user", content: String(prompt ?? "") }],
      }),
      signal: ac.signal,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`OpenAI error ${res.status}: ${body.slice(0, 500)}`);
    }

    const json = await res.json();
    return {
      text: json?.choices?.[0]?.message?.content ?? "",
      provider: "openai",
      usage: json?.usage || null,
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function runLlm(
  prompt,
  { model, timeoutMs = 60_000, caller = "unknown", skipLog = false } = {},
) {
  const started = Date.now();

  if (isAnthropicModel(model)) {
    const result = await runAnthropicAgentPrompt({
      model,
      prompt,
      timeoutMs,
      caller,
      maxTurns: 1,
      skipLog,
    });
    return { text: result.text, durationMs: Date.now() - started };
  }

  const provider = detectModelProvider(model) || "openai";

  try {
    const result = await runOpenAiPrompt({ model, prompt, timeoutMs });
    const durationMs = Date.now() - started;

    if (!skipLog) {
      logLlmCall({
        provider,
        model: model || "gpt-4o-mini",
        caller,
        prompt,
        response: result.text,
        input_tokens:
          result.usage?.prompt_tokens ?? estimateTokensFromChars(String(prompt || "").length),
        output_tokens:
          result.usage?.completion_tokens ??
          estimateTokensFromChars(String(result.text || "").length),
        duration_ms: durationMs,
        ok: true,
      });
    }

    return { text: result.text, durationMs };
  } catch (error) {
    const durationMs = Date.now() - started;
    if (!skipLog) {
      logLlmCall({
        provider,
        model: model || "unknown",
        caller,
        prompt,
        response: "",
        duration_ms: durationMs,
        ok: false,
        error: error?.message || String(error),
      });
    }
    throw error;
  }
}
