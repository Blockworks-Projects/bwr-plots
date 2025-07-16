/**
 * Layout Container
 * ---
 * bwr-plots/frontend/src/components/layout/v2/LayoutContainer.tsx
 * ---
 * Root container component with resize observer and layout management
 */

'use client';

import React, { useRef, useEffect, useState, createContext, useContext } from 'react';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface LayoutContainerProps {
  children: React.ReactNode;
  className?: string;
  minHeight?: number;
  maxHeight?: number;
  onResize?: (dimensions: LayoutDimensions) => void;
}

interface LayoutDimensions {
  width: number;
  height: number;
  availableWidth: number;
  availableHeight: number;
}

interface LayoutContainerContext {
  dimensions: LayoutDimensions;
  isResizing: boolean;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Context                                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

const LayoutContainerContext = createContext<LayoutContainerContext | null>(null);

export function useLayoutContainer() {
  const context = useContext(LayoutContainerContext);
  if (!context) {
    throw new Error('useLayoutContainer must be used within LayoutContainer');
  }
  return context;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Layout Container Component                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function LayoutContainer({
  children,
  className = '',
  minHeight,
  maxHeight,
  onResize
}: LayoutContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState<LayoutDimensions>({
    width: 0,
    height: 0,
    availableWidth: 0,
    availableHeight: 0
  });
  const [isResizing, setIsResizing] = useState(false);
  const resizeTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Resize Observer                                                                    │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        const rect = entry.target.getBoundingClientRect();
        
        const newDimensions: LayoutDimensions = {
          width,
          height,
          availableWidth: width,
          availableHeight: height
        };

        setDimensions(newDimensions);
        onResize?.(newDimensions);

        // Set resizing flag with debounce
        setIsResizing(true);
        if (resizeTimeoutRef.current) {
          clearTimeout(resizeTimeoutRef.current);
        }
        resizeTimeoutRef.current = setTimeout(() => {
          setIsResizing(false);
        }, 150);
      }
    });

    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }
    };
  }, [onResize]);

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Style Computation                                                                  │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  const containerStyle: React.CSSProperties = {
    minHeight: minHeight ? `${minHeight}px` : undefined,
    maxHeight: maxHeight ? `${maxHeight}px` : undefined,
    position: 'relative',
    width: '100%',
    height: '100%'
  };

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Context Value                                                                      │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  const contextValue: LayoutContainerContext = {
    dimensions,
    isResizing,
    containerRef
  };

  return (
    <LayoutContainerContext.Provider value={contextValue}>
      <div
        ref={containerRef}
        className={`layout-container ${className}`}
        style={containerStyle}
        data-resizing={isResizing}
      >
        {children}
        
        {/* Debug overlay in development */}
        {process.env.NODE_ENV === 'development' && (
          <div className="absolute top-0 left-0 z-50 pointer-events-none">
            <div className="bg-black/75 text-white text-xs p-1 rounded-br">
              {dimensions.width} × {dimensions.height}
            </div>
          </div>
        )}
      </div>
    </LayoutContainerContext.Provider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utility Hooks                                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useContainerDimensions() {
  const { dimensions } = useLayoutContainer();
  return dimensions;
}

export function useIsResizing() {
  const { isResizing } = useLayoutContainer();
  return isResizing;
}