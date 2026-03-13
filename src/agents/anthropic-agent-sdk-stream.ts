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

/**
 * Build a pi-ai AssistantMessage from the SDK result text.
 */
function buildAssistantMessageFromSDK(
  text: string,
  modelInfo: { api: string; provider: string; id: string },
  stopReason: StopReason = "stop",
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
    stopReason,
    api: modelInfo.api,
    provider: modelInfo.provider,
    model: modelInfo.id,
    usage,
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
}): StreamFn {
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();

    const run = async () => {
      type SDKEvent = { type: string; subtype?: string; [k: string]: unknown };
      type SDKQuery = AsyncGenerator<SDKEvent, void> & {
        close?: () => void;
        interrupt?: () => Promise<void>;
      };
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
            // Disable all built-in Claude Code tools (Bash, Read, Edit, etc.) — only
            // MCP-bridged OpenClaw tools are available via mcpServers below.
            tools: [] as const,
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

        // Wire up abort signal — uses interrupt() to preserve progress.
        // On abort, the SDK session is saved to disk and can be resumed
        // on the next query via the `resume` option.
        const signal = options?.signal;
        if (signal) {
          const onAbort = async () => {
            log.info("Agent SDK: abort signal received, interrupting subprocess");
            interrupted = true;
            if (sdkStream.interrupt) {
              try {
                await sdkStream.interrupt();
              } catch {
                // Interrupt failed (e.g. subprocess already exited), fall back to close
                sdkStream.close?.();
              }
            } else {
              sdkStream.close?.();
            }
          };
          if (signal.aborted) {
            interrupted = true;
            sdkStream.close?.();
            throw new Error("aborted");
          }
          signal.addEventListener("abort", () => void onAbort(), { once: true });
        }

        // 5. Iterate stream, emit text_delta events for live streaming
        const modelInfo = { api: model.api, provider: model.provider, id: model.id };
        let resultText = "";
        let textBlockStarted = false;
        let gotFinalResult = false;

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

          // SDKResultSuccess event contains the final response text
          if (evt.type === "result" && evt.subtype === "success") {
            resultText = (evt as { result?: string }).result ?? "";
            gotFinalResult = true;
          }
          if (evt.type === "result" && evt.subtype !== "success") {
            // SDKResultError subtypes: error_during_execution | error_max_turns |
            // error_max_budget_usd | error_max_structured_output_retries
            const errEvt = evt as {
              subtype?: string;
              errors?: string[];
              num_turns?: number;
              stop_reason?: string | null;
            };
            const errMsg = `Agent SDK result error: subtype=${errEvt.subtype} turns=${errEvt.num_turns} stop_reason=${errEvt.stop_reason} errors=${(errEvt.errors ?? []).join("; ")}`;
            log.error(errMsg);
            throw new Error(errMsg);
          }

          // SDKPartialAssistantMessage: { type: 'stream_event', event: BetaRawMessageStreamEvent }
          // Extract text deltas from content_block_delta events.
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
          // arrived mid-run, interrupt gracefully so pi-agent-core can process it.
          if (opts?.hasSteeringMessages?.() && !interrupted && !gotFinalResult) {
            log.info("Agent SDK: steering message detected, interrupting for handoff");
            interrupted = true;
            if (sdkStream.interrupt) {
              try {
                await sdkStream.interrupt();
              } catch {
                sdkStream.close?.();
              }
            } else {
              sdkStream.close?.();
            }
            // Loop will exit when the SDK stream ends after interrupt.
          }
        }

        // 6. Close text block and push done/error event
        const wasAborted = options?.signal?.aborted === true;
        const wasInterrupted = interrupted;

        // Save SDK session ID for resume if interrupted (not hard-aborted via /stop).
        if (wasInterrupted && !wasAborted && sdkSessionId) {
          lastSdkSessionId = sdkSessionId;
          log.info(`Agent SDK: saved session ${sdkSessionId} for resume`);
        }

        const stopReason: StopReason = wasAborted ? ("aborted" as StopReason) : "stop";
        const finalText = isSilentReplyText(resultText.trim()) ? "" : resultText.trim();
        const message = buildAssistantMessageFromSDK(finalText, modelInfo, stopReason);

        if (textBlockStarted) {
          stream.push({
            type: "text_end",
            contentIndex: 0,
            content: finalText,
            partial: message,
          });
        }

        log.info(
          `Agent SDK result: model=${model.id} text_len=${resultText.length} aborted=${wasAborted} interrupted=${wasInterrupted}`,
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
