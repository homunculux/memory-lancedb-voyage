/**
 * Embedding Provider Factory
 * Creates the correct embedder implementation based on config.
 */

import type { EmbedderConfig, IEmbedder } from "./embedder-interface.js";
import { VoyageEmbedder } from "./embedder.js";
import { OpenAIEmbedder } from "./embedder-openai.js";
import { JinaEmbedder } from "./embedder-jina.js";

export function createEmbedderFromConfig(config: EmbedderConfig): IEmbedder {
  const embeddingConfig = {
    apiKey: config.apiKey,
    model: config.model,
    dimensions: config.dimensions,
    baseUrl: config.baseUrl,
  };

  switch (config.provider) {
    case "voyage":
      return new VoyageEmbedder(embeddingConfig);
    case "openai":
      return new OpenAIEmbedder(embeddingConfig);
    case "jina":
      return new JinaEmbedder(embeddingConfig);
    default:
      throw new Error(
        `Unknown embedding provider: ${config.provider}. Supported providers: voyage, openai, jina`,
      );
  }
}
