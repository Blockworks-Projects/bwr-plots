/**
 * Resize Observer Hook
 * ---
 * bwr-plots/frontend/src/components/layout/hooks/useResizeObserver.ts
 * ---
 * Hook for observing element resize events with performance optimizations
 */

import { useEffect, useRef, useCallback } from 'react';
import { ResizeEntry, UseResizeObserverOptions } from '../types';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ useResizeObserver Hook                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useResizeObserver({ 
  ref, 
  onResize, 
  debounceMs = 100 
}: UseResizeObserverOptions) {
  const observerRef = useRef<ResizeObserver | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const rafRef = useRef<number | null>(null);

  // Debounced resize handler
  const debouncedResize = useCallback((entry: ResizeEntry) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }

      rafRef.current = requestAnimationFrame(() => {
        onResize?.(entry);
      });
    }, debounceMs);
  }, [onResize, debounceMs]);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    // Create ResizeObserver
    observerRef.current = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        
        debouncedResize({
          width,
          height,
          target: entry.target,
        });
      }
    });

    // Start observing
    observerRef.current.observe(element);

    // Cleanup
    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [ref, debouncedResize]);

  return observerRef.current;
}