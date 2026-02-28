/**
 * Bridge: OpenClaw AgentTool[] → Agent SDK MCP server.
 *
 * Converts TypeBox-based tool schemas to Zod shapes and wraps each tool's
 * execute() as an MCP handler, then registers them via createSdkMcpServer().
 */
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { z } from "zod";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { splitMediaFromOutput } from "../media/parse.js";

const log = createSubsystemLogger("agent-sdk-mcp-bridge");

// ── TypeBox JSON Schema → Zod conversion ────────────────────────────────────

type JsonSchema = Record<string, unknown>;

/**
 * Convert a single JSON Schema property to a Zod type.
 * TypeBox schemas compile to standard JSON Schema objects.
 */
function jsonSchemaPropertyToZod(schema: JsonSchema): z.ZodType {
  const enumValues = schema.enum as string[] | undefined;
  if (enumValues && Array.isArray(enumValues)) {
    // String enums
    if (enumValues.length > 0) {
      return z.enum(enumValues as [string, ...string[]]);
    }
    return z.string();
  }

  const type = schema.type as string | undefined;
  switch (type) {
    case "string":
      return z.string();
    case "number":
    case "integer":
      return z.number();
    case "boolean":
      return z.boolean();
    case "array": {
      const items = schema.items as JsonSchema | undefined;
      return z.array(items ? jsonSchemaPropertyToZod(items) : z.unknown());
    }
    case "object": {
      const props = schema.properties as Record<string, JsonSchema> | undefined;
      if (props) {
        const nested = buildZodShape(props, (schema.required as string[]) ?? []);
        return z.object(nested);
      }
      return z.record(z.string(), z.unknown());
    }
    default:
      return z.unknown();
  }
}

/**
 * Build a Zod raw shape from JSON Schema properties + required array.
 */
function buildZodShape(
  properties: Record<string, JsonSchema>,
  required: string[],
): Record<string, z.ZodType> {
  const requiredSet = new Set(required);
  const shape: Record<string, z.ZodType> = {};

  for (const [key, propSchema] of Object.entries(properties)) {
    const zodType = jsonSchemaPropertyToZod(propSchema);
    shape[key] = requiredSet.has(key) ? zodType : z.optional(zodType);
  }

  return shape;
}

/**
 * Convert a TypeBox tool schema (which IS a JSON Schema object) to a Zod raw shape.
 */
function typeboxSchemaToZodShape(schema: unknown): Record<string, z.ZodType> {
  if (!schema || typeof schema !== "object") {
    return {};
  }
  const s = schema as JsonSchema;
  const properties = (s.properties ?? {}) as Record<string, JsonSchema>;
  const required = (s.required ?? []) as string[];
  return buildZodShape(properties, required);
}

// ── MCP server factory ──────────────────────────────────────────────────────

type McpCallToolResult = {
  content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
  isError?: boolean;
};

type OnToolResultFn = (payload: { text?: string; mediaUrls?: string[] }) => void | Promise<void>;

/**
 * Create an in-process MCP server exposing OpenClaw tools to the Agent SDK.
 *
 * @param tools - OpenClaw AgentTool instances
 * @param sdk - The dynamically-imported @anthropic-ai/claude-agent-sdk module
 */
export function createOpenClawMcpServer(
  tools: AgentTool[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- SDK is dynamically imported
  sdk: Record<string, any>,
  onToolResult?: OnToolResultFn,
): unknown {
  const createSdkMcpServer = sdk.createSdkMcpServer as (options: {
    name: string;
    tools: unknown[];
  }) => unknown;
  const sdkTool = sdk.tool as (
    name: string,
    description: string,
    inputSchema: Record<string, z.ZodType>,
    handler: (args: Record<string, unknown>, extra: unknown) => Promise<McpCallToolResult>,
  ) => unknown;

  const mcpTools = tools.map((agentTool) => {
    const shape = typeboxSchemaToZodShape(agentTool.parameters);
    const toolName = agentTool.name || "unknown";

    return sdkTool(toolName, agentTool.description ?? "", shape, async (args) => {
      const toolCallId = `mcp-${toolName}-${Date.now()}`;
      try {
        const result = await agentTool.execute(toolCallId, args);
        // Deliver media side effects (e.g. TTS audio) directly via callback
        // since Agent SDK tool results don't flow through the normal event pipeline.
        if (onToolResult) {
          for (const part of result?.content ?? []) {
            if (part.type === "text" && part.text) {
              const parsed = splitMediaFromOutput(part.text);
              if (parsed.mediaUrls?.length) {
                void onToolResult({ mediaUrls: parsed.mediaUrls });
              }
            }
          }
        }
        const content = (result?.content ?? []).map(
          (part: { type: string; text?: string; data?: string; mimeType?: string }) => {
            if (part.type === "text") {
              return { type: "text" as const, text: part.text ?? "" };
            }
            if (part.type === "image") {
              return {
                type: "image" as const,
                data: part.data ?? "",
                mimeType: part.mimeType ?? "image/png",
              };
            }
            return { type: "text" as const, text: JSON.stringify(part) };
          },
        );
        return { content };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log.error(`MCP tool ${toolName} error: ${message}`);
        return {
          content: [{ type: "text" as const, text: `Error: ${message}` }],
          isError: true,
        };
      }
    });
  });

  log.info(`MCP bridge: registered ${mcpTools.length} openclaw tools`);
  return createSdkMcpServer({ name: "openclaw", tools: mcpTools });
}
