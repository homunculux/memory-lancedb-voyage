/**
 * Normalize a base URL for LLM endpoints:
 * - Trim whitespace
 * - Strip trailing `/v1` or `/v1/`
 * - Strip trailing slashes
 */
export function normalizeBaseUrl(url: string): string {
  let u = url.trim().replace(/\/+$/, "");
  if (u.toLowerCase().endsWith("/v1")) u = u.slice(0, -3);
  return u.replace(/\/+$/, "");
}

/**
 * Extract host:port from a URL string.
 * Returns null if the URL is unparseable.
 */
export function getUrlHost(url: string): string | null {
  try {
    return new URL(url).host;
  } catch {
    if (!url.includes("://")) {
      try { return new URL(`http://${url}`).host; } catch { /* fall through */ }
    }
    return null;
  }
}
