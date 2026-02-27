import fs from "node:fs";
import path from "node:path";
import { estimateTokensFromChars, logLlmCall } from "./interaction-store.js";
import { normalizeAnthropicModel } from "./model-utils.js";

let smokeTestPromise = null;

function readDotEnvToken() {
  const envPath = path.join(process.cwd(), ".env");
  if (!fs.existsSync(envPath)) {
    return null;
  }
  const raw = fs.readFileSync(envPath, "utf8");
  const m = raw.match(/^\s*CLAUDE_CODE_OAUTH_TOKEN\s*=\s*(.+)\s*$/m);
  if (!m) {
    return null;
  }
  return m[1].trim().replace(/^['"]|['"]$/g, "");
}

export function resolveOAuthTokenOrThrow() {
  if (process.env.ANTHROPIC_API_KEY) {
    throw new Error(
      "OAuth-only mode conflict: ANTHROPIC_API_KEY is set. Remove it and use CLAUDE_CODE_OAUTH_TOKEN only.",
    );
  }

  const token = process.env.CLAUDE_CODE_OAUTH_TOKEN || readDotEnvToken();
  if (!token) {
    throw new Error(
      "Missing Claude OAuth token. Set CLAUDE_CODE_OAUTH_TOKEN (or add it to .env) after running `claude login`.",
    );
  }

  process.env.CLAUDE_CODE_OAUTH_TOKEN = token;
  return token;
}

async function loadQuery() {
  const sdk = await import("@anthropic-ai/claude-agent-sdk");
  const query = sdk.query || sdk.default?.query;
  if (typeof query !== "function") {
    throw new Error("Could not find query() in @anthropic-ai/claude-agent-sdk exports.");
  }
  return query;
}

async function runSingleQuery({ model, prompt, timeoutMs = 20_000, maxTurns = 1 }) {
  resolveOAuthTokenOrThrow();
  const query = await loadQuery();

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(new Error("Anthropic request timed out")), timeoutMs);

  try {
    const stream = query({ model, prompt, tools: [], maxTurns, signal: ac.signal });

    let text = "";
    for await (const evt of stream) {
      const chunks = [
        evt?.text,
        evt?.delta?.text,
        evt?.message?.text,
        evt?.content_block?.text,
      ].filter(Boolean);
      if (chunks.length) {
        text += chunks.join("");
      }

      const blocks = [
        ...(Array.isArray(evt?.content) ? evt.content : []),
        ...(Array.isArray(evt?.message?.content) ? evt.message.content : []),
        ...(evt?.content_block ? [evt.content_block] : []),
      ];
      for (const block of blocks) {
        if ((block?.type === "text" || block?.type === "output_text") && block?.text) {
          text += block.text;
        }
      }
    }

    return text.trim();
  } finally {
    clearTimeout(timer);
  }
}

async function runStartupSmokeTest(model) {
  if (process.env.CLAUDE_OAUTH_SMOKE_TEST_DISABLE === "1") {
    return;
  }

  const text = await runSingleQuery({
    model,
    prompt: "Reply with exactly AUTH_OK and nothing else.",
    timeoutMs: 20_000,
    maxTurns: 1,
  });

  if (!/AUTH_OK/.test(text)) {
    throw new Error(
      `Anthropic OAuth smoke test failed: expected AUTH_OK but got ${JSON.stringify(text).slice(0, 200)}. Credentials may be invalid.`,
    );
  }
}

async function ensureSmokeTest(model) {
  if (!smokeTestPromise) {
    smokeTestPromise = runStartupSmokeTest(model).catch((err) => {
      smokeTestPromise = null;
      throw err;
    });
  }
  return smokeTestPromise;
}

export async function runAnthropicAgentPrompt({
  model,
  prompt,
  timeoutMs = 60_000,
  caller = "unknown",
  maxTurns = 1,
  skipLog = false,
} = {}) {
  const started = Date.now();
  const resolvedModel = normalizeAnthropicModel(model || "claude-opus-4-6");

  try {
    await ensureSmokeTest(resolvedModel);
    const text = await runSingleQuery({
      model: resolvedModel,
      prompt,
      timeoutMs,
      maxTurns: Math.max(1, maxTurns || 1),
    });

    if (!skipLog) {
      logLlmCall({
        provider: "anthropic",
        model: resolvedModel,
        caller,
        prompt,
        response: text,
        input_tokens: estimateTokensFromChars(String(prompt || "").length),
        output_tokens: estimateTokensFromChars(String(text || "").length),
        duration_ms: Date.now() - started,
        ok: true,
      });
    }

    return { text, provider: "anthropic" };
  } catch (error) {
    if (!skipLog) {
      logLlmCall({
        provider: "anthropic",
        model: resolvedModel,
        caller,
        prompt,
        response: "",
        duration_ms: Date.now() - started,
        ok: false,
        error: error?.message || String(error),
      });
    }
    throw error;
  }
}
