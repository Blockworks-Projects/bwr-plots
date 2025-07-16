/**
 * Scroll Position Hook
 * ---
 * bwr-plots/frontend/src/components/layout/hooks/useScrollPosition.ts
 * ---
 * Hook for tracking scroll position with performance optimizations
 */

import { useEffect, useState, useRef } from 'react';
import { UseScrollPositionOptions } from '../types';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Types                                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface ScrollPosition {
  x: number;
  y: number;
  direction: 'up' | 'down' | 'left' | 'right' | null;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ useScrollPosition Hook                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useScrollPosition({
  ref,
  throttleMs = 16,
}: UseScrollPositionOptions = {}) {
  const [scrollPosition, setScrollPosition] = useState<ScrollPosition>({
    x: 0,
    y: 0,
    direction: null,
  });

  const lastScrollRef = useRef({ x: 0, y: 0 });
  const throttleTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const element = ref?.current || window;
    const isWindow = element === window;

    const handleScroll = () => {
      if (throttleTimeoutRef.current) return;

      throttleTimeoutRef.current = setTimeout(() => {
        if (rafRef.current) {
          cancelAnimationFrame(rafRef.current);
        }

        rafRef.current = requestAnimationFrame(() => {
          const currentX = isWindow 
            ? window.scrollX 
            : (element as HTMLElement).scrollLeft;
          const currentY = isWindow 
            ? window.scrollY 
            : (element as HTMLElement).scrollTop;

          const lastX = lastScrollRef.current.x;
          const lastY = lastScrollRef.current.y;

          let direction: ScrollPosition['direction'] = null;
          
          if (currentY > lastY) {
            direction = 'down';
          } else if (currentY < lastY) {
            direction = 'up';
          } else if (currentX > lastX) {
            direction = 'right';
          } else if (currentX < lastX) {
            direction = 'left';
          }

          lastScrollRef.current = { x: currentX, y: currentY };

          setScrollPosition({
            x: currentX,
            y: currentY,
            direction,
          });

          throttleTimeoutRef.current = null;
        });
      }, throttleMs);
    };

    // Initial scroll position
    const initialX = isWindow 
      ? window.scrollX 
      : (element as HTMLElement).scrollLeft;
    const initialY = isWindow 
      ? window.scrollY 
      : (element as HTMLElement).scrollTop;

    lastScrollRef.current = { x: initialX, y: initialY };
    setScrollPosition({
      x: initialX,
      y: initialY,
      direction: null,
    });

    // Add scroll listener
    element.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      element.removeEventListener('scroll', handleScroll);
      
      if (throttleTimeoutRef.current) {
        clearTimeout(throttleTimeoutRef.current);
        throttleTimeoutRef.current = null;
      }
      
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [ref, throttleMs]);

  return scrollPosition;
}