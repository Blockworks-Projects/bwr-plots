/**
 * Scroll Area
 * ---
 * bwr-plots/frontend/src/components/layout/primitives/ScrollArea.tsx
 * ---
 * Performance-optimized scrollable container with optional virtualization
 */

'use client';

import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import { ScrollAreaProps, ScrollEvent } from '../types';
import styles from '../styles/layout.module.css';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

const SCROLLBAR_HIDE_DELAY = 1000;
const SCROLL_THROTTLE_MS = 16; // ~60fps

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Virtual Scroller Hook                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

function useVirtualScroller(
  containerRef: React.RefObject<HTMLDivElement | null>,
  options?: ScrollAreaProps['virtualizer']
) {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 0 });

  useEffect(() => {
    if (!options || !containerRef.current) return;

    const container = containerRef.current;
    const { itemCount, itemSize, overscan = 3, horizontal = false } = options;

    const calculateVisibleRange = () => {
      const scrollPos = horizontal ? container.scrollLeft : container.scrollTop;
      const containerSize = horizontal ? container.clientWidth : container.clientHeight;

      if (typeof itemSize === 'number') {
        const start = Math.max(0, Math.floor(scrollPos / itemSize) - overscan);
        const end = Math.min(
          itemCount,
          Math.ceil((scrollPos + containerSize) / itemSize) + overscan
        );
        setVisibleRange({ start, end });
      } else {
        // Variable size items - more complex calculation
        let accumulatedSize = 0;
        let start = 0;
        let end = itemCount;

        for (let i = 0; i < itemCount; i++) {
          const size = itemSize(i);
          if (accumulatedSize + size > scrollPos && start === 0) {
            start = Math.max(0, i - overscan);
          }
          if (accumulatedSize > scrollPos + containerSize) {
            end = Math.min(itemCount, i + overscan);
            break;
          }
          accumulatedSize += size;
        }

        setVisibleRange({ start, end });
      }
    };

    calculateVisibleRange();

    const handleScroll = () => {
      calculateVisibleRange();
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [containerRef, options]);

  return visibleRange;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ ScrollArea Component                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ScrollArea({
  children,
  height,
  width,
  maxHeight,
  maxWidth,
  scrollbarWidth = 'auto',
  scrollHideDelay = SCROLLBAR_HIDE_DELAY,
  onScroll,
  virtualizer,
  className = '',
  style,
}: ScrollAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [showScrollbar, setShowScrollbar] = useState(false);
  const hideTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const scrollRafRef = useRef<number | undefined>(undefined);
  const visibleRange = useVirtualScroller(scrollRef, virtualizer);

  // Handle scroll events with throttling
  const handleScroll = useCallback(() => {
    if (scrollRafRef.current) cancelAnimationFrame(scrollRafRef.current);

    scrollRafRef.current = requestAnimationFrame(() => {
      if (!scrollRef.current) return;

      const scrollEvent: ScrollEvent = {
        scrollTop: scrollRef.current.scrollTop,
        scrollLeft: scrollRef.current.scrollLeft,
        scrollHeight: scrollRef.current.scrollHeight,
        scrollWidth: scrollRef.current.scrollWidth,
        clientHeight: scrollRef.current.clientHeight,
        clientWidth: scrollRef.current.clientWidth,
      };

      onScroll?.(scrollEvent);

      // Show scrollbar
      setShowScrollbar(true);

      // Hide scrollbar after delay
      if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = setTimeout(() => {
        setShowScrollbar(false);
      }, scrollHideDelay);
    });
  }, [onScroll, scrollHideDelay]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
      if (scrollRafRef.current) cancelAnimationFrame(scrollRafRef.current);
    };
  }, []);

  // Calculate container styles
  const containerStyles = useMemo(() => ({
    height,
    width,
    maxHeight,
    maxWidth,
    ...style,
  }), [height, width, maxHeight, maxWidth, style]);

  // Render virtualized content
  const renderContent = () => {
    if (virtualizer) {
      const { itemCount, itemSize, horizontal = false } = virtualizer;
      const { start, end } = visibleRange;

      // Calculate total size
      let totalSize = 0;
      if (typeof itemSize === 'number') {
        totalSize = itemCount * itemSize;
      } else {
        for (let i = 0; i < itemCount; i++) {
          totalSize += itemSize(i);
        }
      }

      // Calculate offset for visible items
      let offset = 0;
      if (typeof itemSize === 'number') {
        offset = start * itemSize;
      } else {
        for (let i = 0; i < start; i++) {
          offset += itemSize(i);
        }
      }

      return (
        <div
          style={{
            [horizontal ? 'width' : 'height']: `${totalSize}px`,
            position: 'relative',
          }}
        >
          <div
            style={{
              transform: horizontal 
                ? `translateX(${offset}px)` 
                : `translateY(${offset}px)`,
              position: 'absolute',
              top: 0,
              left: 0,
            }}
          >
            {React.Children.toArray(children).slice(start, end)}
          </div>
        </div>
      );
    }

    return children;
  };

  // Determine scrollbar classes
  const scrollAreaClasses = [
    styles.scrollArea,
    scrollbarWidth === 'thin' && styles.scrollbarThin,
    scrollbarWidth === 'none' && styles.scrollbarNone,
    showScrollbar && styles.scrollbarVisible,
    className,
  ].filter(Boolean).join(' ');

  return (
    <div
      ref={scrollRef}
      className={scrollAreaClasses}
      style={containerStyles}
      onScroll={handleScroll}
      data-testid="scroll-area"
    >
      {renderContent()}
    </div>
  );
}