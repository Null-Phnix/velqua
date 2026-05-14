/**
 * API communication layer.
 * Provides the base URL and a fetch wrapper with error extraction.
 */

export const API_BASE: string = window.location.origin;

/**
 * Fetch wrapper that throws on non-ok responses.
 * Extracts error detail from JSON body when available.
 */
export async function apiFetch(url: string, opts?: RequestInit): Promise<Response> {
    const response = await fetch(url, opts);
    if (!response.ok) {
        let detail = response.statusText;
        try {
            const err = await response.json() as { detail?: string };
            detail = err.detail ?? detail;
        } catch (_) {}
        throw new Error(detail);
    }
    return response;
}
