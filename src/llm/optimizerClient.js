import fs from "fs";
import OpenAI from "openai";
import { SYSTEM_PROMPT } from "./prompt.js";
import { LLM_RESPONSE_FORMAT } from "./responseSchema.js";
import { buildModelOptions } from "./modelOptions.js";
import {
  parseLlmResponse,
  InvalidLlmResponseError,
} from "./responseValidator.js";

const MAX_COMPLETION_TOKENS = 8192;

/**
 * Call an OpenAI-compatible chat completion endpoint to get an optimization plan.
 *
 * When `config.mockResponsePath` is set, reads the response JSON from disk
 * instead of making a network request (used in tests).
 *
 * @param {object} options
 * @param {object} options.payload Request payload from `buildLlmRequestPayload`.
 * @param {object} options.config Runtime config from `loadConfig`.
 * @param {{ debug?: (...args: any[]) => void } | undefined} options.logger
 * @returns {Promise<{ version: number, changes: any[], globalRationale: any, warnings: string[] }>}
 */
export async function requestOptimization({ payload, config, logger }) {
  if (config.mockResponsePath) {
    const mock = fs.readFileSync(config.mockResponsePath, "utf8");
    logger?.debug?.(
      `Using mock LLM response from file: ${config.mockResponsePath}`,
    );
    return parseLlmResponse(mock);
  }

  if (!config.apiKey) {
    throw new Error("API key missing. Set OPENAI_API_KEY (or pass --api-key).");
  }

  const client = new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseURL,
  });

  const targetBase = config.baseURL ?? "https://api.openai.com/v1";
  logger?.debug?.(`LLM target ${targetBase} | model=${config.model}`);
  logger?.debug?.("System prompt:\n", SYSTEM_PROMPT.trim());
  logger?.debug?.(
    "User payload message:\n",
    JSON.stringify(redactImages(payload), null, 2),
  );

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    buildUserMessage(payload),
  ];

  const content = await callChatCompletion({
    client,
    config,
    messages,
    logger,
  });
  try {
    return parseLlmResponse(content);
  } catch (error) {
    if (error instanceof InvalidLlmResponseError) {
      logger?.debug?.(`Invalid LLM response; raw content:\n${content}`);
    }
    throw error;
  }
}

async function callChatCompletion({ client, config, messages, logger }) {
  const formatError = (error) => {
    if (error?.response) {
      const body = JSON.stringify(error.response.data ?? {}, null, 2);
      return `status=${error.response.status} data=${body}`;
    }
    return error?.message ?? "unknown error";
  };

  logger?.debug?.("Sending chat completion with structured schema.");
  logger?.debug?.(
    "Response schema:\n",
    JSON.stringify(LLM_RESPONSE_FORMAT, null, 2),
  );
  try {
    const completion = await client.chat.completions.create({
      ...buildModelOptions(config),
      max_completion_tokens: MAX_COMPLETION_TOKENS,
      response_format: LLM_RESPONSE_FORMAT,
      messages,
    });
    const choice = completion.choices?.[0];
    const finishReason = choice?.finish_reason ?? completion.stop_reason;
    if (["length", "max_tokens"].includes(finishReason)) {
      throw new Error(
        `LLM response exceeded the ${MAX_COMPLETION_TOKENS}-token output limit. Reduce the request size or use a model with a larger output limit.`,
      );
    }
    const content = extractTextContent(
      choice?.message?.content ?? completion.content,
    );
    if (!content.trim()) {
      const refusal = choice?.message?.refusal;
      if (typeof refusal === "string" && refusal.trim()) {
        throw new Error(`LLM refused the request: ${refusal}`);
      }
      throw new Error(buildEmptyContentMessage(config, finishReason));
    }
    /* c8 ignore next */
    logger?.debug?.("LLM raw response:", content);
    return content;
  } catch (error) {
    logger?.debug?.(`LLM request failed: ${formatError(error)}`);
    throw new Error(`LLM request failed: ${formatError(error)}`);
  }
}

function extractTextContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (part) =>
        part &&
        typeof part === "object" &&
        (part.type === "text" || typeof part.text === "string") &&
        typeof part.text === "string",
    )
    .map((part) => part.text)
    .join("");
}

function buildEmptyContentMessage(config, finishReason) {
  const finishDetail = finishReason ? ` (finish_reason=${finishReason})` : "";
  if (/^https?:\/\/(?:www\.)?chutes\.ai(?:\/|$)/i.test(config.baseURL ?? "")) {
    return "LLM returned no text content. Chutes API requests must use https://llm.chutes.ai/v1, not https://chutes.ai. Also choose a model ID from https://llm.chutes.ai/v1/models; claude-opus-5-5 is not in Chutes' public model catalog.";
  }
  return `LLM returned no text content${finishDetail}. Verify the configured base URL and that the selected model is available from that provider.`;
}

function buildUserMessage(payload) {
  const {
    projectSummary,
    currentSettings,
    intentDetails,
    userModifiedSettings,
    allowUserSettingOverrides,
    targetLanguage,
  } = payload;
  const dataForModel = {
    version: payload.version,
    projectSummary,
    currentSettings,
    userModifiedSettings: userModifiedSettings ?? [],
    allowUserSettingOverrides: allowUserSettingOverrides === true,
    targetLanguage: targetLanguage ?? "en",
  };
  const intentLines = buildIntentLines(intentDetails, {
    targetLanguage: dataForModel.targetLanguage,
    allowUserSettingOverrides: dataForModel.allowUserSettingOverrides,
    userModifiedSettings: dataForModel.userModifiedSettings,
  });

  const content = [
    ...(intentLines.length
      ? [
          {
            type: "text",
            text: `Intent details (explicit):\n${intentLines.join("\n")}`,
          },
        ]
      : []),
    {
      type: "text",
      text: `Structured project data (JSON):\n${JSON.stringify(dataForModel, null, 2)}`,
    },
  ];

  (payload.plateImages ?? []).forEach((image) => {
    content.push({
      type: "text",
      text: `Plate preview: ${image.name}${Number.isFinite(image.plateIndex) ? ` (plate index ${image.plateIndex})` : ""}`,
    });
    content.push({
      type: "image_url",
      image_url: { url: image.dataUrl, detail: "low" },
    });
  });

  return { role: "user", content };
}

function buildIntentLines(intentDetails, extras = {}) {
  const lines = [];
  pushIfNonEmptyString(lines, "primary_goal", intentDetails?.primary_goal);
  pushIfNonEmptyArray(lines, "secondary_goals", intentDetails?.secondary_goals);
  pushIfNonEmptyArray(lines, "preferred_focus", intentDetails?.preferred_focus);
  pushConstraintLines(lines, intentDetails?.constraints);
  pushIfNonEmptyArray(
    lines,
    "locked_parameters",
    intentDetails?.locked_parameters,
  );
  pushIfTrue(lines, "load_bearing", intentDetails?.load_bearing);
  pushIfTrue(lines, "safety_critical", intentDetails?.safety_critical);
  pushIfNonEmptyString(
    lines,
    "free_text_description",
    intentDetails?.free_text_description,
  );
  pushIfNonEmptyString(lines, "targetLanguage", extras.targetLanguage);
  if (extras.allowUserSettingOverrides) {
    lines.push("allowUserSettingOverrides: true");
  }
  if (
    Array.isArray(extras.userModifiedSettings) &&
    extras.userModifiedSettings.length > 0
  ) {
    lines.push(
      `userModifiedSettings: ${extras.userModifiedSettings.join(", ")}`,
    );
  }
  return lines;
}

function pushIfNonEmptyString(lines, key, value) {
  if (typeof value !== "string") return;
  const trimmed = value.trim();
  if (!trimmed) return;
  lines.push(`${key}: ${trimmed}`);
}

function pushIfNonEmptyArray(lines, key, values) {
  if (!Array.isArray(values) || values.length === 0) return;
  lines.push(`${key}: ${values.join(", ")}`);
}

function pushIfTrue(lines, key, value) {
  if (value !== true) return;
  lines.push(`${key}: true`);
}

function pushConstraintLines(lines, constraints) {
  const maxTime = constraints?.max_print_time_hours;
  if (
    typeof maxTime === "number" ||
    (typeof maxTime === "string" && maxTime.trim())
  ) {
    lines.push(`constraints.max_print_time_hours: ${maxTime}`);
  }
  if (constraints?.material_saving_important === true) {
    lines.push("constraints.material_saving_important: true");
  }
}

function redactImages(payload) {
  if (!payload?.plateImages?.length) return payload;
  return {
    ...payload,
    plateImages: payload.plateImages.map((img) => ({
      ...img,
      dataUrl: `[image ${img.name} :: ${img.dataUrl?.length ?? 0} chars]`,
    })),
  };
}
