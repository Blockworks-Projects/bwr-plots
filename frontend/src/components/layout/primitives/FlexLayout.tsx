/**
 * Flex Layout
 * ---
 * bwr-plots/frontend/src/components/layout/primitives/FlexLayout.tsx
 * ---
 * Flexible layout system with optional resizable regions
 */

'use client';

import React, { Children, cloneElement, useRef, useState, useCallback, useMemo } from 'react';
import { FlexLayoutProps, FlexItemProps } from '../types';
import styles from '../styles/layout.module.css';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

const MIN_PANEL_SIZE = 100;
const RESIZE_HANDLE_SIZE = 8;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ FlexItem Component                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function FlexItem({
  children,
  size,
  grow = 0,
  shrink = 1,
  basis = 'auto',
  minSize,
  maxSize,
  className = '',
  style,
}: FlexItemProps) {
  const flexStyles = useMemo(() => ({
    flex: size ? `0 0 ${size}` : `${grow} ${shrink} ${basis}`,
    minWidth: minSize,
    maxWidth: maxSize,
    minHeight: minSize,
    maxHeight: maxSize,
    ...style,
  }), [size, grow, shrink, basis, minSize, maxSize, style]);

  return (
    <div 
      className={`${styles.flexItem} ${className}`}
      style={flexStyles}
    >
      {children}
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ ResizeHandle Component                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface ResizeHandleProps {
  orientation: 'horizontal' | 'vertical';
  onResize: (delta: number) => void;
  onResizeEnd: () => void;
}

function ResizeHandle({ orientation, onResize, onResizeEnd }: ResizeHandleProps) {
  const [isDragging, setIsDragging] = useState(false);
  const startPosRef = useRef<number>(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startPosRef.current = orientation === 'horizontal' ? e.clientX : e.clientY;

    const handleMouseMove = (e: MouseEvent) => {
      const currentPos = orientation === 'horizontal' ? e.clientX : e.clientY;
      const delta = currentPos - startPosRef.current;
      startPosRef.current = currentPos;
      onResize(delta);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      onResizeEnd();
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [orientation, onResize, onResizeEnd]);

  const handleClasses = [
    styles.resizeHandle,
    styles[`resizeHandle${orientation === 'horizontal' ? 'Vertical' : 'Horizontal'}`],
    isDragging && styles.resizeHandleDragging,
  ].filter(Boolean).join(' ');

  return (
    <div
      className={handleClasses}
      onMouseDown={handleMouseDown}
      role="separator"
      aria-orientation={orientation}
      aria-label="Resize handle"
      tabIndex={0}
    />
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ FlexLayout Component                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function FlexLayout({
  children,
  orientation = 'horizontal',
  gap = 0,
  wrap = false,
  align = 'stretch',
  justify = 'start',
  resizable = false,
  onResize,
  className = '',
  style,
}: FlexLayoutProps) {
  const childrenArray = Children.toArray(children) as React.ReactElement<FlexItemProps>[];
  const [sizes, setSizes] = useState<number[]>(() => 
    childrenArray.map(() => 0)
  );
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle resize for a specific panel
  const handleResize = useCallback((index: number, delta: number) => {
    setSizes((prevSizes) => {
      const newSizes = [...prevSizes];
      const currentSize = newSizes[index];
      const nextSize = newSizes[index + 1];

      if (currentSize + delta >= MIN_PANEL_SIZE && nextSize - delta >= MIN_PANEL_SIZE) {
        newSizes[index] = currentSize + delta;
        newSizes[index + 1] = nextSize - delta;
      }

      return newSizes;
    });
  }, []);

  const handleResizeEnd = useCallback(() => {
    if (onResize) {
      onResize(sizes);
    }
  }, [sizes, onResize]);

  // Calculate flex layout styles
  const layoutStyles = useMemo(() => ({
    display: 'flex',
    flexDirection: orientation === 'horizontal' ? 'row' : 'column',
    flexWrap: wrap ? 'wrap' : 'nowrap',
    alignItems: align,
    justifyContent: justify,
    gap: `${gap}px`,
    ...style,
  } as React.CSSProperties), [orientation, wrap, align, justify, gap, style]);

  // Render children with resize handles
  const renderChildren = () => {
    return childrenArray.reduce((acc: React.ReactNode[], child, index) => {
      // Clone child with size if resizable
      const childWithSize = resizable && sizes[index] > 0
        ? cloneElement(child, {
            size: `${sizes[index]}px`,
          })
        : child;

      acc.push(
        <React.Fragment key={index}>
          {childWithSize}
          {resizable && index < childrenArray.length - 1 && (
            <ResizeHandle
              orientation={orientation}
              onResize={(delta) => handleResize(index, delta)}
              onResizeEnd={handleResizeEnd}
            />
          )}
        </React.Fragment>
      );

      return acc;
    }, []);
  };

  // Initialize sizes on mount
  React.useEffect(() => {
    if (resizable && containerRef.current) {
      const container = containerRef.current;
      const totalSize = orientation === 'horizontal' 
        ? container.clientWidth 
        : container.clientHeight;
      
      const childCount = childrenArray.length;
      const totalGap = gap * (childCount - 1);
      const totalHandles = RESIZE_HANDLE_SIZE * (childCount - 1);
      const availableSize = totalSize - totalGap - totalHandles;
      const defaultSize = availableSize / childCount;

      setSizes(childrenArray.map(() => defaultSize));
    }
  }, [resizable, orientation, gap, childrenArray.length]);

  return (
    <div
      ref={containerRef}
      className={`${styles.flexLayout} ${className}`}
      style={layoutStyles}
    >
      {renderChildren()}
    </div>
  );
}