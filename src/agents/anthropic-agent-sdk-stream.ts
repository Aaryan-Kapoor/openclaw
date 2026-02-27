import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, StopReason, Usage } from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("anthropic-agent-sdk-stream");

// ── Message formatting ──────────────────────────────────────────────────────

type InputContentPart =
  | { type: "text"; text: string }
  | { type: "image"; data: string }
  | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return (content as InputContentPart[])
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}

/**
 * Convert pi-ai messages into a single prompt string for the Agent SDK.
 *
 * The SDK's `query()` accepts a single `prompt` string, not a messages array.
 * We pass the system prompt separately via `options.systemPrompt`, so here we
 * only format the conversation history — most importantly the latest user message.
 *
 * For context, we include recent history in a lightweight format so the SDK
 * subprocess can see the conversation thread.
 */
function formatConversationForSDK(messages: Array<{ role: string; content: unknown }>): string {
  // Find the last user message — this is the primary prompt.
  let lastUserText = "";
  const historyParts: string[] = [];

  for (const msg of messages) {
    const text = extractTextContent(msg.content).trim();
    if (!text) {
      continue;
    }

    if (msg.role === "user") {
      lastUserText = text;
    } else if (msg.role === "assistant") {
      // Accumulate history context
      historyParts.push(`[Assistant]: ${text}`);
    }
  }

  // If there's meaningful history, prepend it for context
  if (historyParts.length > 0 && lastUserText) {
    // Keep only the most recent history to stay within reasonable prompt size
    const recentHistory = historyParts.slice(-5).join("\n\n");
    return `<conversation_context>\n${recentHistory}\n</conversation_context>\n\n${lastUserText}`;
  }

  return lastUserText || "Hello";
}

/**
 * Build a pi-ai AssistantMessage from the SDK result text.
 */
function buildAssistantMessageFromSDK(
  text: string,
  modelInfo: { api: string; provider: string; id: string },
): AssistantMessage {
  const usage: Usage = {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };

  return {
    role: "assistant",
    content: text ? [{ type: "text", text }] : [],
    stopReason: "stop" as StopReason,
    api: modelInfo.api,
    provider: modelInfo.provider,
    model: modelInfo.id,
    usage,
    timestamp: Date.now(),
  };
}

// ── Main StreamFn factory ───────────────────────────────────────────────────

export function createAnthropicAgentSDKStreamFn(): StreamFn {
  return (model, context, _options) => {
    const stream = createAssistantMessageEventStream();

    const run = async () => {
      try {
        // 1. Load the Agent SDK dynamically
        const sdk = await import("@anthropic-ai/claude-agent-sdk");
        const queryFn =
          sdk.query ??
          (sdk as unknown as { default?: { query?: typeof sdk.query } }).default?.query;
        if (typeof queryFn !== "function") {
          throw new Error("Could not find query() in @anthropic-ai/claude-agent-sdk exports.");
        }

        // 2. Format conversation history + latest message into prompt
        const prompt = formatConversationForSDK(
          (context.messages ?? []) as Array<{ role: string; content: unknown }>,
        );

        // 3. Call Agent SDK with system prompt + tools
        log.info(
          `Agent SDK call: model=${model.id} prompt_len=${prompt.length} system_len=${(context.systemPrompt ?? "").length}`,
        );

        const sdkStream = queryFn({
          prompt,
          options: {
            model: model.id,
            systemPrompt: context.systemPrompt || undefined,
            tools: { type: "preset", preset: "claude_code" },
            maxTurns: 10,
            permissionMode: "acceptEdits",
          },
        });

        // 4. Iterate stream, collect result
        let resultText = "";
        for await (const evt of sdkStream) {
          // SDKResultSuccess event contains the final response text
          if (evt.type === "result" && evt.subtype === "success") {
            resultText = (evt as { result?: string }).result ?? "";
          }
          // Also capture streaming text chunks
          const chunks = [
            (evt as { text?: string }).text,
            (evt as { delta?: { text?: string } }).delta?.text,
            (evt as { message?: { text?: string } }).message?.text,
            (evt as { content_block?: { text?: string } }).content_block?.text,
          ].filter(Boolean);
          if (chunks.length && !resultText) {
            resultText += chunks.join("");
          }
          // Check content blocks
          const blocks = [
            ...(Array.isArray((evt as { content?: unknown[] }).content)
              ? ((evt as { content: unknown[] }).content as Array<{ type?: string; text?: string }>)
              : []),
            ...(Array.isArray((evt as { message?: { content?: unknown[] } }).message?.content)
              ? ((evt as { message: { content: unknown[] } }).message.content as Array<{
                  type?: string;
                  text?: string;
                }>)
              : []),
            ...((evt as { content_block?: { type?: string; text?: string } }).content_block
              ? [(evt as { content_block: { type?: string; text?: string } }).content_block]
              : []),
          ];
          for (const block of blocks) {
            if (
              (block.type === "text" || block.type === "output_text") &&
              block.text &&
              !resultText
            ) {
              resultText += block.text;
            }
          }
        }

        // 5. Build AssistantMessage and push done event
        const message = buildAssistantMessageFromSDK(resultText.trim(), {
          api: model.api,
          provider: model.provider,
          id: model.id,
        });

        log.info(`Agent SDK result: model=${model.id} text_len=${resultText.length}`);

        stream.push({
          type: "done",
          reason: "stop",
          message,
        });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        log.error(`Agent SDK error: ${errorMessage}`);
        stream.push({
          type: "error",
          reason: "error",
          error: {
            role: "assistant" as const,
            content: [],
            stopReason: "error" as StopReason,
            errorMessage,
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            timestamp: Date.now(),
          },
        });
      } finally {
        stream.end();
      }
    };

    queueMicrotask(() => void run());
    return stream;
  };
}
