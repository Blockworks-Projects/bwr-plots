/**
 * Performance Utilities
 * ---
 * bwr-plots/frontend/src/components/layout/utils/performance.ts
 * ---
 * Performance optimization utilities for layout components
 */

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Animation Frame Utilities                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Request animation frame with fallback
 */
export function requestAnimationFrame(callback: FrameRequestCallback): number {
  if (typeof window !== 'undefined' && window.requestAnimationFrame) {
    return window.requestAnimationFrame(callback);
  }
  
  // Fallback for environments without requestAnimationFrame
  return setTimeout(callback, 16) as unknown as number;
}

/**
 * Cancel animation frame with fallback
 */
export function cancelAnimationFrame(id: number): void {
  if (typeof window !== 'undefined' && window.cancelAnimationFrame) {
    window.cancelAnimationFrame(id);
  } else {
    clearTimeout(id);
  }
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Throttling and Debouncing                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Throttle function calls using requestAnimationFrame
 */
export function throttleRAF<T extends (...args: any[]) => void>(
  func: T,
  immediate = false
): T {
  let rafId: number | null = null;
  let lastArgs: Parameters<T> | null = null;
  
  const throttled = ((...args: Parameters<T>) => {
    lastArgs = args;
    
    if (rafId === null) {
      if (immediate) {
        func(...args);
      }
      
      rafId = requestAnimationFrame(() => {
        if (!immediate && lastArgs) {
          func(...lastArgs);
        }
        rafId = null;
        lastArgs = null;
      });
    }
  }) as T;
  
  // Add cancel method
  (throttled as any).cancel = () => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
      lastArgs = null;
    }
  };
  
  return throttled;
}

/**
 * Debounce function calls
 */
export function debounce<T extends (...args: any[]) => void>(
  func: T,
  wait: number,
  immediate = false
): T {
  let timeout: NodeJS.Timeout | null = null;
  
  const debounced = ((...args: Parameters<T>) => {
    const later = () => {
      timeout = null;
      if (!immediate) func(...args);
    };
    
    const callNow = immediate && !timeout;
    
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    
    if (callNow) func(...args);
  }) as T;
  
  // Add cancel method
  (debounced as any).cancel = () => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }
  };
  
  return debounced;
}

/**
 * Throttle function calls
 */
export function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number
): T {
  let inThrottle: boolean = false;
  let lastFunc: NodeJS.Timeout | null = null;
  let lastRan: number = 0;
  
  const throttled = ((...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      lastRan = Date.now();
      inThrottle = true;
    } else {
      if (lastFunc) clearTimeout(lastFunc);
      lastFunc = setTimeout(() => {
        if (Date.now() - lastRan >= limit) {
          func(...args);
          lastRan = Date.now();
        }
      }, limit - (Date.now() - lastRan));
    }
  }) as T;
  
  // Add cancel method
  (throttled as any).cancel = () => {
    if (lastFunc) {
      clearTimeout(lastFunc);
      lastFunc = null;
    }
    inThrottle = false;
  };
  
  return throttled;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Intersection Observer Utilities                                                    │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Create an intersection observer with default options
 */
export function createIntersectionObserver(
  callback: IntersectionObserverCallback,
  options?: IntersectionObserverInit
): IntersectionObserver {
  const defaultOptions: IntersectionObserverInit = {
    root: null,
    rootMargin: '0px',
    threshold: 0,
    ...options,
  };
  
  return new IntersectionObserver(callback, defaultOptions);
}

/**
 * Observe element visibility with callback
 */
export function observeElementVisibility(
  element: Element,
  callback: (isVisible: boolean, entry: IntersectionObserverEntry) => void,
  options?: IntersectionObserverInit
): () => void {
  const observer = createIntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        callback(entry.isIntersecting, entry);
      });
    },
    options
  );
  
  observer.observe(element);
  
  return () => observer.disconnect();
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Resize Observer Utilities                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Create a resize observer with default options
 */
export function createResizeObserver(
  callback: ResizeObserverCallback,
  options?: ResizeObserverOptions
): ResizeObserver {
  return new ResizeObserver(callback);
}

/**
 * Observe element size changes with callback
 */
export function observeElementSize(
  element: Element,
  callback: (size: { width: number; height: number }) => void,
  debounceMs = 100
): () => void {
  const debouncedCallback = debounce(callback, debounceMs);
  
  const observer = createResizeObserver((entries) => {
    entries.forEach((entry) => {
      const { width, height } = entry.contentRect;
      debouncedCallback({ width, height });
    });
  });
  
  observer.observe(element);
  
  return () => {
    observer.disconnect();
    (debouncedCallback as any).cancel();
  };
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Memory Management                                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Weak reference map for caching
 */
export class WeakCache<K extends object, V> {
  private cache = new WeakMap<K, V>();
  
  get(key: K): V | undefined {
    return this.cache.get(key);
  }
  
  set(key: K, value: V): void {
    this.cache.set(key, value);
  }
  
  has(key: K): boolean {
    return this.cache.has(key);
  }
  
  delete(key: K): boolean {
    return this.cache.delete(key);
  }
}

/**
 * LRU cache implementation
 */
export class LRUCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;
  
  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }
  
  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }
  
  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      // Update existing key
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove least recently used
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    
    this.cache.set(key, value);
  }
  
  has(key: K): boolean {
    return this.cache.has(key);
  }
  
  delete(key: K): boolean {
    return this.cache.delete(key);
  }
  
  clear(): void {
    this.cache.clear();
  }
  
  get size(): number {
    return this.cache.size;
  }
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ DOM Utilities                                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Check if element supports CSS containment
 */
export function supportsCSSContainment(): boolean {
  if (typeof CSS === 'undefined') return false;
  return CSS.supports('contain', 'layout style paint');
}

/**
 * Apply CSS containment to element
 */
export function applyCSSContainment(element: HTMLElement, types: string[] = ['layout', 'style', 'paint']): void {
  if (supportsCSSContainment()) {
    element.style.contain = types.join(' ');
  }
}

/**
 * Check if element supports CSS content-visibility
 */
export function supportsContentVisibility(): boolean {
  if (typeof CSS === 'undefined') return false;
  return CSS.supports('content-visibility', 'auto');
}

/**
 * Apply content-visibility to element
 */
export function applyContentVisibility(element: HTMLElement, value: 'auto' | 'hidden' | 'visible' = 'auto'): void {
  if (supportsContentVisibility()) {
    (element.style as any).contentVisibility = value;
  }
}