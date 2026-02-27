import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

const MAX_TEXT_LEN = 10_000;

export const MODEL_PRICING_PER_1M = {
  "claude-opus-4": { input: 15, output: 75 },
  "claude-opus-4-6": { input: 15, output: 75 },
  "claude-sonnet-4": { input: 3, output: 15 },
  "claude-sonnet-4-6": { input: 3, output: 15 },
  "claude-3-5-sonnet-latest": { input: 3, output: 15 },
  "claude-3-5-haiku-latest": { input: 0.8, output: 4 },
  "gpt-4.1": { input: 5, output: 15 },
  "gpt-4.1-mini": { input: 0.6, output: 2.4 },
  "gpt-4o": { input: 5, output: 15 },
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "gpt-5": { input: 10, output: 30 },
};

let db;
let insertStmt;

function dbPath() {
  const out = process.env.LLM_CALLS_DB_PATH || path.join(process.cwd(), "data", "llm-calls.sqlite");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  return out;
}

function getDb() {
  if (db && insertStmt) {
    return db;
  }
  db = new DatabaseSync(dbPath());
  db.exec("PRAGMA journal_mode = WAL;");
  db.exec(`
    CREATE TABLE IF NOT EXISTS llm_calls (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      provider TEXT,
      model TEXT,
      caller TEXT,
      prompt TEXT,
      response TEXT,
      input_tokens INTEGER,
      output_tokens INTEGER,
      cost_estimate REAL,
      duration_ms INTEGER,
      ok INTEGER NOT NULL DEFAULT 1,
      error TEXT
    );
  `);
  insertStmt = db.prepare(`
    INSERT INTO llm_calls (
      timestamp, provider, model, caller, prompt, response,
      input_tokens, output_tokens, cost_estimate, duration_ms, ok, error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  return db;
}

function truncate(value, maxLen = MAX_TEXT_LEN) {
  if (value == null) {
    return null;
  }
  const s = String(value);
  return s.length <= maxLen ? s : `${s.slice(0, maxLen)}…[truncated]`;
}

export function redactSecrets(value) {
  if (!value) {
    return value;
  }
  let text = String(value);
  text = text.replace(/\b(sk-[a-zA-Z0-9_-]{12,})\b/g, "[REDACTED_API_KEY]");
  text = text.replace(
    /\b(ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)\s*=\s*[^\s"']+/gi,
    "$1=[REDACTED]",
  );
  text = text.replace(/\bBearer\s+[A-Za-z0-9._\-+/=]+/gi, "Bearer [REDACTED_TOKEN]");
  return text;
}

export function estimateTokensFromChars(charCount) {
  if (!charCount || charCount <= 0) {
    return 0;
  }
  return Math.ceil(charCount / 4);
}

function selectPriceModel(model = "") {
  const lower = String(model).toLowerCase();
  if (MODEL_PRICING_PER_1M[lower]) {
    return lower;
  }
  if (lower.includes("opus-4-6") || lower.includes("opus-4.6")) {
    return "claude-opus-4-6";
  }
  if (lower.includes("opus-4")) {
    return "claude-opus-4";
  }
  if (lower.includes("sonnet-4-6") || lower.includes("sonnet-4.6")) {
    return "claude-sonnet-4-6";
  }
  if (lower.includes("sonnet-4")) {
    return "claude-sonnet-4";
  }
  if (lower.includes("haiku")) {
    return "claude-3-5-haiku-latest";
  }
  if (lower.includes("gpt-4o-mini")) {
    return "gpt-4o-mini";
  }
  if (lower.includes("gpt-4o")) {
    return "gpt-4o";
  }
  if (lower.includes("gpt-4.1-mini")) {
    return "gpt-4.1-mini";
  }
  if (lower.includes("gpt-4.1")) {
    return "gpt-4.1";
  }
  if (lower.includes("gpt-5")) {
    return "gpt-5";
  }
  return null;
}

export function estimateCostUsd({ model, inputTokens = 0, outputTokens = 0 }) {
  const key = selectPriceModel(model);
  const p = key ? MODEL_PRICING_PER_1M[key] : null;
  if (!p) {
    return null;
  }
  const inCost = (inputTokens / 1_000_000) * p.input;
  const outCost = (outputTokens / 1_000_000) * p.output;
  return Number((inCost + outCost).toFixed(8));
}

export function logLlmCall(record) {
  setImmediate(() => {
    try {
      getDb();
      const prompt = truncate(redactSecrets(record.prompt));
      const response = truncate(redactSecrets(record.response));
      const error = truncate(redactSecrets(record.error));
      const inputTokens = record.input_tokens ?? estimateTokensFromChars((prompt || "").length);
      const outputTokens = record.output_tokens ?? estimateTokensFromChars((response || "").length);
      const costEstimate =
        record.cost_estimate ?? estimateCostUsd({ model: record.model, inputTokens, outputTokens });

      insertStmt.run(
        new Date().toISOString(),
        record.provider || null,
        record.model || null,
        record.caller || null,
        prompt,
        response,
        inputTokens,
        outputTokens,
        costEstimate,
        Number.isFinite(record.duration_ms) ? Math.round(record.duration_ms) : null,
        record.ok === false ? 0 : 1,
        error || null,
      );
    } catch {
      // fire-and-forget logger by design
    }
  });
}
