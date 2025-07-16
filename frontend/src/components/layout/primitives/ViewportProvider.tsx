/**
 * Viewport Provider
 * ---
 * bwr-plots/frontend/src/components/layout/primitives/ViewportProvider.tsx
 * ---
 * Global viewport state management with performance optimizations
 */

'use client';

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { ViewportState, ViewportProviderProps, Breakpoint } from '../types';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

const DEFAULT_VIEWPORT_STATE: ViewportState = {
  width: 0,
  height: 0,
  scrollX: 0,
  scrollY: 0,
  breakpoint: 'lg',
  isDesktop: true,
  isMobile: false,
  isTablet: false,
};

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Context                                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

const ViewportContext = createContext<ViewportState | undefined>(undefined);

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Helper Functions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

function getBreakpoint(width: number): Breakpoint {
  if (width < BREAKPOINTS.sm) return 'sm';
  if (width < BREAKPOINTS.md) return 'md';
  if (width < BREAKPOINTS.lg) return 'lg';
  if (width < BREAKPOINTS.xl) return 'xl';
  return '2xl';
}

function getDeviceType(width: number) {
  const isMobile = width < BREAKPOINTS.md;
  const isTablet = width >= BREAKPOINTS.md && width < BREAKPOINTS.lg;
  const isDesktop = width >= BREAKPOINTS.lg;
  
  return { isMobile, isTablet, isDesktop };
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ ViewportProvider Component                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ViewportProvider({ children, debounceMs = 100 }: ViewportProviderProps) {
  const [viewport, setViewport] = useState<ViewportState>(DEFAULT_VIEWPORT_STATE);
  const resizeTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const rafRef = useRef<number | undefined>(undefined);

  // Initialize viewport state
  useEffect(() => {
    const updateViewport = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;
      const breakpoint = getBreakpoint(width);
      const deviceType = getDeviceType(width);

      setViewport({
        width,
        height,
        scrollX,
        scrollY,
        breakpoint,
        ...deviceType,
      });
    };

    // Initial update
    updateViewport();

    return () => {
      if (resizeTimeoutRef.current) clearTimeout(resizeTimeoutRef.current);
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Handle resize events
  const handleResize = useCallback(() => {
    if (resizeTimeoutRef.current) clearTimeout(resizeTimeoutRef.current);
    
    resizeTimeoutRef.current = setTimeout(() => {
      rafRef.current = requestAnimationFrame(() => {
        const width = window.innerWidth;
        const height = window.innerHeight;
        const breakpoint = getBreakpoint(width);
        const deviceType = getDeviceType(width);

        setViewport((prev) => ({
          ...prev,
          width,
          height,
          breakpoint,
          ...deviceType,
        }));
      });
    }, debounceMs);
  }, [debounceMs]);

  // Handle scroll events
  const handleScroll = useCallback(() => {
    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    
    scrollTimeoutRef.current = setTimeout(() => {
      rafRef.current = requestAnimationFrame(() => {
        setViewport((prev) => ({
          ...prev,
          scrollX: window.scrollX,
          scrollY: window.scrollY,
        }));
      });
    }, debounceMs / 2); // Scroll updates can be more frequent
  }, [debounceMs]);

  // Set up event listeners
  useEffect(() => {
    window.addEventListener('resize', handleResize, { passive: true });
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('orientationchange', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, [handleResize, handleScroll]);

  return (
    <ViewportContext.Provider value={viewport}>
      {children}
    </ViewportContext.Provider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Hook                                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useViewport() {
  const context = useContext(ViewportContext);
  
  if (!context) {
    throw new Error('useViewport must be used within a ViewportProvider');
  }
  
  return context;
}