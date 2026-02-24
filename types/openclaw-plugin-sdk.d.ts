/**
 * Ambient type declarations for openclaw/plugin-sdk
 * Extracted from the OpenClaw plugin API types needed by this plugin.
 */
declare module "openclaw/plugin-sdk" {
  import type { Command } from "commander";
  import type { TSchema } from "@sinclair/typebox";

  export type PluginLogger = {
    debug?: (message: string) => void;
    info: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
  };

  export type PluginConfigUiHint = {
    label?: string;
    help?: string;
    tags?: string[];
    advanced?: boolean;
    sensitive?: boolean;
    placeholder?: string;
  };

  export type PluginKind = "memory";

  export type PluginConfigValidation =
    | { ok: true; value?: unknown }
    | { ok: false; errors: string[] };

  export type OpenClawPluginConfigSchema = {
    safeParse?: (value: unknown) => {
      success: boolean;
      data?: unknown;
      error?: {
        issues?: Array<{ path: Array<string | number>; message: string }>;
      };
    };
    parse?: (value: unknown) => unknown;
    validate?: (value: unknown) => PluginConfigValidation;
    uiHints?: Record<string, PluginConfigUiHint>;
    jsonSchema?: Record<string, unknown>;
  };

  export type AnyAgentTool = {
    name: string;
    label?: string;
    description: string;
    parameters: TSchema;
    execute: (
      toolCallId: string,
      params: Record<string, unknown>,
    ) => Promise<{
      content: Array<{ type: string; text: string }>;
      details?: Record<string, unknown>;
    }>;
  };

  export type OpenClawPluginToolOptions = {
    name?: string;
    names?: string[];
    optional?: boolean;
  };

  export type OpenClawPluginCliContext = {
    program: Command;
    config: Record<string, unknown>;
    workspaceDir?: string;
    logger: PluginLogger;
  };

  export type OpenClawPluginCliRegistrar = (
    ctx: OpenClawPluginCliContext,
  ) => void | Promise<void>;

  export type OpenClawPluginServiceContext = {
    config: Record<string, unknown>;
    workspaceDir?: string;
    stateDir: string;
    logger: PluginLogger;
  };

  export type OpenClawPluginService = {
    id: string;
    start: (ctx: OpenClawPluginServiceContext) => void | Promise<void>;
    stop?: (ctx: OpenClawPluginServiceContext) => void | Promise<void>;
  };

  export type PluginHookName =
    | "before_model_resolve"
    | "before_prompt_build"
    | "before_agent_start"
    | "llm_input"
    | "llm_output"
    | "agent_end"
    | "before_compaction"
    | "after_compaction"
    | "before_reset"
    | "message_received"
    | "message_sending"
    | "message_sent"
    | "before_tool_call"
    | "after_tool_call"
    | "tool_result_persist"
    | "before_message_write"
    | "session_start"
    | "session_end"
    | "subagent_spawning"
    | "subagent_delivery_target"
    | "subagent_spawned"
    | "subagent_ended"
    | "gateway_start"
    | "gateway_stop";

  export type PluginHookAgentContext = {
    agentId?: string;
    sessionKey?: string;
    sessionId?: string;
    workspaceDir?: string;
    messageProvider?: string;
  };

  export type PluginHookBeforeAgentStartEvent = {
    prompt: string;
    messages?: unknown[];
  };

  export type PluginHookBeforeAgentStartResult = {
    systemPrompt?: string;
    prependContext?: string;
    modelOverride?: string;
    providerOverride?: string;
  };

  export type PluginHookAgentEndEvent = {
    messages: unknown[];
    success: boolean;
    error?: string;
    durationMs?: number;
  };

  export type PluginHookHandlerMap = {
    before_agent_start: (
      event: PluginHookBeforeAgentStartEvent,
      ctx: PluginHookAgentContext,
    ) => Promise<PluginHookBeforeAgentStartResult | void> | PluginHookBeforeAgentStartResult | void;
    agent_end: (
      event: PluginHookAgentEndEvent,
      ctx: PluginHookAgentContext,
    ) => Promise<void> | void;
    [key: string]: (...args: any[]) => any;
  };

  export type OpenClawPluginApi = {
    id: string;
    name: string;
    version?: string;
    description?: string;
    source: string;
    config: Record<string, unknown>;
    pluginConfig?: Record<string, unknown>;
    runtime: Record<string, unknown>;
    logger: PluginLogger;
    registerTool: (
      tool: AnyAgentTool,
      opts?: OpenClawPluginToolOptions,
    ) => void;
    registerHook: (
      events: string | string[],
      handler: (...args: any[]) => any,
      opts?: Record<string, unknown>,
    ) => void;
    registerHttpHandler: (handler: (...args: any[]) => any) => void;
    registerHttpRoute: (params: {
      path: string;
      handler: (...args: any[]) => any;
    }) => void;
    registerChannel: (registration: unknown) => void;
    registerGatewayMethod: (
      method: string,
      handler: (...args: any[]) => any,
    ) => void;
    registerCli: (
      registrar: OpenClawPluginCliRegistrar,
      opts?: { commands?: string[] },
    ) => void;
    registerService: (service: OpenClawPluginService) => void;
    registerProvider: (provider: unknown) => void;
    registerCommand: (command: unknown) => void;
    resolvePath: (input: string) => string;
    on: <K extends PluginHookName>(
      hookName: K,
      handler: K extends keyof PluginHookHandlerMap
        ? PluginHookHandlerMap[K]
        : (...args: any[]) => any,
      opts?: { priority?: number },
    ) => void;
  };

  export type OpenClawPluginDefinition = {
    id?: string;
    name?: string;
    description?: string;
    version?: string;
    kind?: PluginKind;
    configSchema?: OpenClawPluginConfigSchema;
    register?: (api: OpenClawPluginApi) => void | Promise<void>;
    activate?: (api: OpenClawPluginApi) => void | Promise<void>;
  };
}
