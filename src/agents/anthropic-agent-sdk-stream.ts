import type { AgentTool, StreamFn } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, StopReason, Usage } from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { isSilentReplyText } from "../auto-reply/tokens.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { createOpenClawMcpServer } from "./anthropic-agent-sdk-mcp-bridge.js";

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
  const historyParts: string[] = [];
  let lastUserText = "";

  for (const msg of messages) {
    const text = extractTextContent(msg.content).trim();
    if (!text) {
      continue;
    }

    if (msg.role === "user") {
      // Push previous user message to history before overwriting
      if (lastUserText) {
        historyParts.push(`[User]: ${lastUserText}`);
      }
      lastUserText = text;
    } else if (msg.role === "assistant") {
      historyParts.push(`[Assistant]: ${text}`);
    }
  }

  // Prepend recent history for context
  if (historyParts.length > 0 && lastUserText) {
    const recentHistory = historyParts.slice(-10).join("\n\n");
    return `<conversation_context>\n${recentHistory}\n</conversation_context>\n\n${lastUserText}`;
  }

  return lastUserText || "Hello";
}

// ── SDK usage extraction ────────────────────────────────────────────────────

/** Shape of the SDK's NonNullableUsage / BetaUsage on result events. */
type SDKUsage = {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
};

/** Shape of SDK's per-model ModelUsage on result events. */
type SDKModelUsage = {
  inputTokens?: number;
  outputTokens?: number;
  cacheReadInputTokens?: number;
  cacheCreationInputTokens?: number;
  costUSD?: number;
  contextWindow?: number;
};

/**
 * Usage tracker for SDK streams.
 *
 * Pi-agent-core uses AssistantMessage.usage to track context fill and trigger
 * compaction. The SDK's result event carries SESSION-TOTAL usage (all API calls
 * across all turns summed), which is far larger than the actual context window
 * fill. Feeding session totals into the message usage causes false compactions.
 *
 * Instead, we track two things separately:
 * - lastTurnUsage: the most recent SDKAssistantMessage's per-turn token counts.
 *   This represents the actual current context window fill (input_tokens = prompt
 *   size sent to the API on the last turn). This drives context tracking.
 * - costUsd / contextWindow: from the result event, for display only.
 */
interface UsageTracker {
  /** Last assistant turn's usage — represents current context fill. */
  lastInput: number;
  lastOutput: number;
  lastCacheRead: number;
  lastCacheWrite: number;
  /** Session-level cost from result event. */
  costUsd: number;
  /** Context window size from modelUsage. */
  contextWindow: number | undefined;
}

function createUsageTracker(): UsageTracker {
  return {
    lastInput: 0,
    lastOutput: 0,
    lastCacheRead: 0,
    lastCacheWrite: 0,
    costUsd: 0,
    contextWindow: undefined,
  };
}

/** Update with per-turn usage from SDKAssistantMessage (replaces, not accumulates). */
function updateLastTurnUsage(tracker: UsageTracker, messageUsage: SDKUsage | undefined): void {
  if (!messageUsage) {
    return;
  }
  // Each SDKAssistantMessage carries that turn's usage — use the latest one.
  tracker.lastInput = messageUsage.input_tokens ?? 0;
  tracker.lastOutput = messageUsage.output_tokens ?? 0;
  tracker.lastCacheRead = messageUsage.cache_read_input_tokens ?? 0;
  tracker.lastCacheWrite = messageUsage.cache_creation_input_tokens ?? 0;
}

/** Extract cost and contextWindow from SDKResultSuccess/SDKResultError. */
function extractResultMeta(
  tracker: UsageTracker,
  evt: { usage?: SDKUsage; modelUsage?: Record<string, SDKModelUsage>; total_cost_usd?: number },
): void {
  if (typeof evt.total_cost_usd === "number") {
    tracker.costUsd = evt.total_cost_usd;
  }
  if (evt.modelUsage) {
    for (const mu of Object.values(evt.modelUsage)) {
      if (typeof mu.contextWindow === "number" && mu.contextWindow > 0) {
        tracker.contextWindow = mu.contextWindow;
        break;
      }
    }
  }
}

function trackerToUsage(tracker: UsageTracker): Usage {
  const total = tracker.lastInput + tracker.lastOutput;
  return {
    input: tracker.lastInput,
    output: tracker.lastOutput,
    cacheRead: tracker.lastCacheRead,
    cacheWrite: tracker.lastCacheWrite,
    totalTokens: total,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: tracker.costUsd,
    },
  };
}

// ── Message building ────────────────────────────────────────────────────────

function buildAssistantMessageFromSDK(
  text: string,
  modelInfo: { api: string; provider: string; id: string },
  stopReason: StopReason = "stop",
  usage?: Usage,
): AssistantMessage {
  return {
    role: "assistant",
    content: text ? [{ type: "text", text }] : [],
    stopReason,
    api: modelInfo.api,
    provider: modelInfo.provider,
    model: modelInfo.id,
    usage: usage ?? {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    timestamp: Date.now(),
  };
}

// ── SDK session tracking for resume support ─────────────────────────────────

/** Last interrupted SDK session ID, keyed by OpenClaw session (via model+context). */
let lastSdkSessionId: string | undefined;

// ── Main StreamFn factory ───────────────────────────────────────────────────

export function createAnthropicAgentSDKStreamFn(opts?: {
  onToolResult?: (payload: { text?: string; mediaUrls?: string[] }) => void | Promise<void>;
  /** Check if steering messages are queued (new user message arrived mid-run). */
  hasSteeringMessages?: () => boolean;
  /** Called when SDK reports compaction events. */
  onCompaction?: (phase: "start" | "end") => void;
}): StreamFn {
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();

    const run = async () => {
      type SDKEvent = { type: string; subtype?: string; [k: string]: unknown };
      type SDKQuery = AsyncGenerator<SDKEvent, void> & { close?: () => void };
      let sdkStream: SDKQuery;
      let interrupted = false;
      let sdkSessionId: string | undefined;

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

        // 3. Build MCP server from OpenClaw tools (if available)
        const mcpServers: Record<string, unknown> = {};
        const contextTools = (context as { tools?: AgentTool[] }).tools;
        if (Array.isArray(contextTools) && contextTools.length > 0) {
          mcpServers["openclaw"] = createOpenClawMcpServer(contextTools, sdk, opts?.onToolResult);
          log.info(`Agent SDK: ${contextTools.length} openclaw tools registered via MCP`);
        }

        // 4. Call Agent SDK with system prompt + MCP tools
        //    If we have a previous interrupted session, resume it instead of starting fresh.
        const resumeId = lastSdkSessionId;
        if (resumeId) {
          log.info(`Agent SDK: resuming interrupted session ${resumeId}`);
          lastSdkSessionId = undefined;
        }

        log.info(
          `Agent SDK call: model=${model.id} prompt_len=${prompt.length} system_len=${(context.systemPrompt ?? "").length} resume=${resumeId ?? "none"}`,
        );

        sdkStream = queryFn({
          prompt,
          options: {
            model: model.id,
            systemPrompt: context.systemPrompt || undefined,
            // Enable Claude Code's built-in WebFetch and WebSearch (free, no API key).
            // All other built-in tools disabled — the agent uses MCP-bridged
            // OpenClaw tools via mcpServers below instead.
            tools: ["WebFetch", "WebSearch"],
            mcpServers: mcpServers as Record<string, never>,
            // maxTurns omitted — unlimited turns, matching pi-agent-core behavior
            // for all other providers (OpenAI, Google, etc.).
            permissionMode: "bypassPermissions",
            allowDangerouslySkipPermissions: true,
            includePartialMessages: true,
            // Resume a previously interrupted session if available.
            ...(resumeId ? { resume: resumeId } : {}),
          },
        }) as SDKQuery;

        // Wire up abort signal — uses close() to immediately kill the subprocess.
        // Session is preserved on disk (persistSession defaults to true), so
        // the next query() can resume via the `resume` option.
        // close() is used instead of interrupt() because interrupt() is cooperative
        // (waits for subprocess to respond) and hangs when tools are executing.
        const signal = options?.signal;
        if (signal) {
          const onAbort = () => {
            log.info("Agent SDK: abort signal received, closing subprocess");
            interrupted = true;
            sdkStream.close?.();
          };
          if (signal.aborted) {
            interrupted = true;
            sdkStream.close?.();
            throw new Error("aborted");
          }
          signal.addEventListener("abort", onAbort, { once: true });
        }

        // 5. Iterate stream, emit text_delta events for live streaming
        const modelInfo = { api: model.api, provider: model.provider, id: model.id };
        let resultText = "";
        let textBlockStarted = false;
        let gotFinalResult = false;
        const usageTracker = createUsageTracker();
        let isCompacting = false;

        // Partial message built up as deltas arrive
        const partial = buildAssistantMessageFromSDK("", modelInfo);
        stream.push({ type: "start", partial });

        const emitDelta = (delta: string) => {
          if (!delta) {
            return;
          }
          if (!textBlockStarted) {
            textBlockStarted = true;
            partial.content = [{ type: "text", text: "" }];
            stream.push({ type: "text_start", contentIndex: 0, partial });
          }
          const textBlock = partial.content[0] as { type: "text"; text: string };
          textBlock.text += delta;
          stream.push({ type: "text_delta", contentIndex: 0, delta, partial });
        };

        for await (const evt of sdkStream) {
          // Capture SDK session ID from any event for resume support.
          if (!sdkSessionId && typeof evt.session_id === "string") {
            sdkSessionId = evt.session_id;
          }

          // SDKResultSuccess — final response text + session-total usage.
          if (evt.type === "result" && evt.subtype === "success") {
            resultText = (evt as { result?: string }).result ?? "";
            gotFinalResult = true;
            extractResultMeta(
              usageTracker,
              evt as {
                usage?: SDKUsage;
                modelUsage?: Record<string, SDKModelUsage>;
                total_cost_usd?: number;
              },
            );
          }

          // SDKResultError — still carries usage data.
          if (evt.type === "result" && evt.subtype !== "success") {
            const errEvt = evt as {
              subtype?: string;
              errors?: string[];
              num_turns?: number;
              stop_reason?: string | null;
              usage?: SDKUsage;
              modelUsage?: Record<string, SDKModelUsage>;
              total_cost_usd?: number;
            };
            extractResultMeta(usageTracker, errEvt);
            const errMsg = `Agent SDK result error: subtype=${errEvt.subtype} turns=${errEvt.num_turns} stop_reason=${errEvt.stop_reason} errors=${(errEvt.errors ?? []).join("; ")}`;
            log.error(errMsg);
            throw new Error(errMsg);
          }

          // SDKAssistantMessage — per-turn usage from the full BetaMessage.
          if (evt.type === "assistant") {
            const msg = (evt as { message?: { usage?: SDKUsage } }).message;
            updateLastTurnUsage(usageTracker, msg?.usage);
          }

          // SDKStatusMessage — compaction status changes.
          if (evt.type === "system" && evt.subtype === "status") {
            const status = (evt as { status?: string | null }).status;
            if (status === "compacting" && !isCompacting) {
              isCompacting = true;
              opts?.onCompaction?.("start");
              log.info("Agent SDK: compaction started");
            } else if (status !== "compacting" && isCompacting) {
              isCompacting = false;
              opts?.onCompaction?.("end");
              log.info("Agent SDK: compaction ended");
            }
          }

          // SDKCompactBoundaryMessage — explicit compaction boundary.
          if (evt.type === "system" && evt.subtype === "compact_boundary") {
            const meta = (evt as { compact_metadata?: { pre_tokens?: number; trigger?: string } })
              .compact_metadata;
            log.info(
              `Agent SDK: compact boundary trigger=${meta?.trigger} pre_tokens=${meta?.pre_tokens}`,
            );
          }

          // SDKPartialAssistantMessage — extract text deltas for live streaming.
          if (evt.type === "stream_event" && !gotFinalResult) {
            const streamEvt = (evt as { event?: Record<string, unknown> }).event;
            if (
              streamEvt?.type === "content_block_delta" &&
              (streamEvt.delta as Record<string, unknown> | undefined)?.type === "text_delta"
            ) {
              const delta = (streamEvt.delta as { text?: string }).text;
              if (typeof delta === "string") {
                emitDelta(delta);
                resultText += delta;
              }
            }
          }

          // Check for steering messages between SDK events. If a new user message
          // arrived mid-run, close the subprocess so pi-agent-core can process
          // the steering message. Session is saved to disk for resume.
          if (opts?.hasSteeringMessages?.() && !interrupted && !gotFinalResult) {
            log.info("Agent SDK: steering message detected, closing for handoff");
            interrupted = true;
            sdkStream.close?.();
          }
        }

        // 6. Close text block and push done/error event
        const wasAborted = options?.signal?.aborted === true;
        const wasInterrupted = interrupted;

        // Save SDK session ID for resume. close() preserves session on disk
        // (persistSession defaults to true), so both /stop and steering can resume.
        if ((wasInterrupted || wasAborted) && sdkSessionId) {
          lastSdkSessionId = sdkSessionId;
          log.info(`Agent SDK: saved session ${sdkSessionId} for resume`);
        }

        const finalUsage = trackerToUsage(usageTracker);
        const stopReason: StopReason = wasAborted ? ("aborted" as StopReason) : "stop";
        const finalText = isSilentReplyText(resultText.trim()) ? "" : resultText.trim();
        const message = buildAssistantMessageFromSDK(finalText, modelInfo, stopReason, finalUsage);

        if (textBlockStarted) {
          stream.push({
            type: "text_end",
            contentIndex: 0,
            content: finalText,
            partial: message,
          });
        }

        log.info(
          `Agent SDK result: model=${model.id} text_len=${resultText.length} last_turn=${finalUsage.input}in/${finalUsage.output}out cache=${finalUsage.cacheRead}r/${finalUsage.cacheWrite}w cost=$${usageTracker.costUsd.toFixed(4)} aborted=${wasAborted} interrupted=${wasInterrupted}`,
        );

        if (wasAborted) {
          stream.push({
            type: "error",
            reason: "aborted",
            error: message,
          });
        } else {
          stream.push({
            type: "done",
            reason: "stop",
            message,
          });
        }
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
