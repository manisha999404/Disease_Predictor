// Client-side stub for node:async_hooks. The real implementation is
// server-only; this file is only loaded in the client bundle when
// dead-code-elimination is disabled and AsyncLocalStorage leaks in.
export class AsyncLocalStorage<T = unknown> {
  private store: T | undefined;
  run<R>(store: T, callback: () => R): R {
    this.store = store;
    return callback();
  }
  getStore(): T | undefined {
    return this.store;
  }
  enterWith(store: T): void {
    this.store = store;
  }
  exit<R>(callback: () => R): R {
    return callback();
  }
}

export default { AsyncLocalStorage };