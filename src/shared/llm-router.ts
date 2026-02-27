export type RunLlmOptions = {
  model?: string;
  timeoutMs?: number;
  caller?: string;
  skipLog?: boolean;
};

export type RunLlmResult = {
  text: string;
  durationMs: number;
};

export async function runLlm(prompt: string, options: RunLlmOptions = {}): Promise<RunLlmResult> {
  // @ts-expect-error runtime JS module intentionally loaded from repo-level shared/
  const mod = await import("../../shared/llm-router.js");
  return mod.runLlm(prompt, options);
}
