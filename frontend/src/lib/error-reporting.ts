/**
 * Generic client-side error reporting utility.
 * Wire up captureException to your own error tracking service
 * (e.g. Sentry, Datadog, Highlight.io) if needed.
 */

type ErrorOptions = {
  mechanism?: "manual" | "onerror" | "unhandledrejection" | "react_error_boundary";
  handled?: boolean;
  severity?: "error" | "warning" | "info";
};

type ErrorEvents = {
  captureException?: (
    error: unknown,
    context?: Record<string, unknown>,
    options?: ErrorOptions,
  ) => void;
};

declare global {
  interface Window {
    __errorEvents?: ErrorEvents;
  }
}

export function reportError(error: unknown, context: Record<string, unknown> = {}) {
  if (typeof window === "undefined") return;

  // Replace the line below with your own error-tracking SDK call, e.g.:
  //   Sentry.captureException(error, { extra: { ...context } });
  window.__errorEvents?.captureException?.(
    error,
    {
      source: "react_error_boundary",
      route: window.location.pathname,
      ...context,
    },
    {
      mechanism: "react_error_boundary",
      handled: false,
      severity: "error",
    },
  );
}
