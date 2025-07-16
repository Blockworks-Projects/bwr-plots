/**
 * Scroll Area
 * ---
 * bwr-plots/frontend/src/components/layout/v2/ScrollArea.tsx
 * ---
 * Managed scrolling component with position persistence
 */

'use client';

import React, { useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import { cn } from '@/lib/utils';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface ScrollAreaProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onScroll'> {
  orientation?: 'vertical' | 'horizontal' | 'both';
  persistKey?: string;
  onScrollPositionChange?: (position: ScrollPosition) => void;
  restorePosition?: boolean;
  showScrollbar?: 'always' | 'auto' | 'hover';
  smoothScroll?: boolean;
}

export interface ScrollPosition {
  top: number;
  left: number;
  scrollHeight: number;
  scrollWidth: number;
  clientHeight: number;
  clientWidth: number;
  scrollPercentageX: number;
  scrollPercentageY: number;
}

export interface ScrollAreaHandle {
  scrollTo: (options: ScrollToOptions) => void;
  scrollBy: (options: ScrollToOptions) => void;
  getScrollPosition: () => ScrollPosition;
  saveScrollPosition: () => void;
  restoreScrollPosition: () => void;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Scroll Position Storage                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

const scrollPositionCache = new Map<string, { top: number; left: number }>();

function saveScrollPosition(key: string, position: { top: number; left: number }) {
  scrollPositionCache.set(key, position);
  
  // Also save to sessionStorage for persistence across hot reloads
  if (typeof window !== 'undefined') {
    try {
      const stored = JSON.parse(sessionStorage.getItem('scroll-positions') || '{}');
      stored[key] = position;
      sessionStorage.setItem('scroll-positions', JSON.stringify(stored));
    } catch (e) {
      // Ignore storage errors
    }
  }
}

function getScrollPosition(key: string): { top: number; left: number } | null {
  // Check memory cache first
  const cached = scrollPositionCache.get(key);
  if (cached) return cached;
  
  // Check sessionStorage
  if (typeof window !== 'undefined') {
    try {
      const stored = JSON.parse(sessionStorage.getItem('scroll-positions') || '{}');
      return stored[key] || null;
    } catch (e) {
      return null;
    }
  }
  
  return null;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Scroll Area Component                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const ScrollArea = forwardRef<ScrollAreaHandle, ScrollAreaProps>(
  (
    {
      orientation = 'vertical',
      persistKey,
      onScrollPositionChange,
      restorePosition = true,
      showScrollbar = 'auto',
      smoothScroll = false,
      className,
      style,
      children,
      ...props
    },
    ref
  ) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const lastScrollPosition = useRef<ScrollPosition | null>(null);
    const isRestoringPosition = useRef(false);

    // ┌────────────────────────────────────────────────────────────────────────────────────┐
    // │ Scroll Utilities                                                                   │
    // └────────────────────────────────────────────────────────────────────────────────────┘

    const getScrollPositionData = (): ScrollPosition => {
      const element = scrollRef.current;
      if (!element) {
        return {
          top: 0,
          left: 0,
          scrollHeight: 0,
          scrollWidth: 0,
          clientHeight: 0,
          clientWidth: 0,
          scrollPercentageX: 0,
          scrollPercentageY: 0
        };
      }

      const scrollPercentageX = element.scrollWidth > element.clientWidth
        ? (element.scrollLeft / (element.scrollWidth - element.clientWidth)) * 100
        : 0;
      
      const scrollPercentageY = element.scrollHeight > element.clientHeight
        ? (element.scrollTop / (element.scrollHeight - element.clientHeight)) * 100
        : 0;

      return {
        top: element.scrollTop,
        left: element.scrollLeft,
        scrollHeight: element.scrollHeight,
        scrollWidth: element.scrollWidth,
        clientHeight: element.clientHeight,
        clientWidth: element.clientWidth,
        scrollPercentageX,
        scrollPercentageY
      };
    };

    const savePosition = () => {
      if (persistKey && !isRestoringPosition.current) {
        const position = getScrollPositionData();
        saveScrollPosition(persistKey, { top: position.top, left: position.left });
      }
    };

    const restorePositionFn = () => {
      if (!persistKey || !restorePosition || !scrollRef.current) return;
      
      const saved = getScrollPosition(persistKey);
      if (saved) {
        isRestoringPosition.current = true;
        scrollRef.current.scrollTop = saved.top;
        scrollRef.current.scrollLeft = saved.left;
        
        // Reset flag after a delay
        setTimeout(() => {
          isRestoringPosition.current = false;
        }, 100);
      }
    };

    // ┌────────────────────────────────────────────────────────────────────────────────────┐
    // │ Event Handlers                                                                     │
    // └────────────────────────────────────────────────────────────────────────────────────┘

    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
      const position = getScrollPositionData();
      lastScrollPosition.current = position;
      
      // Save position if we have a persist key
      if (persistKey && !isRestoringPosition.current) {
        savePosition();
      }
      
      // Call user's onScroll handler
      onScrollPositionChange?.(position);
    };

    // ┌────────────────────────────────────────────────────────────────────────────────────┐
    // │ Imperative Handle                                                                  │
    // └────────────────────────────────────────────────────────────────────────────────────┘

    useImperativeHandle(ref, () => ({
      scrollTo: (options: ScrollToOptions) => {
        scrollRef.current?.scrollTo(options);
      },
      scrollBy: (options: ScrollToOptions) => {
        scrollRef.current?.scrollBy(options);
      },
      getScrollPosition: getScrollPositionData,
      saveScrollPosition: savePosition,
      restoreScrollPosition: restorePositionFn
    }));

    // ┌────────────────────────────────────────────────────────────────────────────────────┐
    // │ Effects                                                                            │
    // └────────────────────────────────────────────────────────────────────────────────────┘

    // Restore scroll position on mount
    useEffect(() => {
      if (restorePosition) {
        // Delay to ensure content is rendered
        const timer = setTimeout(restorePositionFn, 50);
        return () => clearTimeout(timer);
      }
    }, []);

    // Save scroll position before unmount
    useEffect(() => {
      return () => {
        if (persistKey && lastScrollPosition.current) {
          saveScrollPosition(persistKey, {
            top: lastScrollPosition.current.top,
            left: lastScrollPosition.current.left
          });
        }
      };
    }, [persistKey]);

    // ┌────────────────────────────────────────────────────────────────────────────────────┐
    // │ Style Classes                                                                      │
    // └────────────────────────────────────────────────────────────────────────────────────┘

    const orientationClasses = {
      vertical: 'overflow-y-auto overflow-x-hidden',
      horizontal: 'overflow-x-auto overflow-y-hidden',
      both: 'overflow-auto'
    };

    const scrollbarClasses = {
      always: 'scrollbar-always',
      auto: '',
      hover: 'scrollbar-hover'
    };

    return (
      <div
        ref={scrollRef}
        className={cn(
          'scroll-area',
          orientationClasses[orientation],
          scrollbarClasses[showScrollbar],
          smoothScroll && 'scroll-smooth',
          className
        )}
        style={style}
        onScroll={handleScroll}
        {...props}
      >
        {children}
      </div>
    );
  }
);

ScrollArea.displayName = 'ScrollArea';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Scroll Observer Hook                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useScrollObserver(
  ref: React.RefObject<HTMLElement>,
  callback: (position: ScrollPosition) => void,
  deps: React.DependencyList = []
) {
  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const handleScroll = () => {
      const position: ScrollPosition = {
        top: element.scrollTop,
        left: element.scrollLeft,
        scrollHeight: element.scrollHeight,
        scrollWidth: element.scrollWidth,
        clientHeight: element.clientHeight,
        clientWidth: element.clientWidth,
        scrollPercentageX: element.scrollWidth > element.clientWidth
          ? (element.scrollLeft / (element.scrollWidth - element.clientWidth)) * 100
          : 0,
        scrollPercentageY: element.scrollHeight > element.clientHeight
          ? (element.scrollTop / (element.scrollHeight - element.clientHeight)) * 100
          : 0
      };
      
      callback(position);
    };

    element.addEventListener('scroll', handleScroll, { passive: true });
    
    // Call once on mount
    handleScroll();

    return () => {
      element.removeEventListener('scroll', handleScroll);
    };
  }, [ref, ...deps]);
}