/**
 * Panel
 * ---
 * bwr-plots/frontend/src/components/layout/primitives/Panel.tsx
 * ---
 * Base container component with header, actions, and collapsible support
 */

'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import { PanelProps } from '../types';
import styles from '../styles/layout.module.css';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

const DEFAULT_PADDING = 16;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Component                                                                    │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function Panel({
  children,
  title,
  description,
  actions,
  padding = true,
  bordered = true,
  elevated = false,
  collapsible = false,
  defaultCollapsed = false,
  onCollapse,
  className = '',
  style,
}: PanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  // Handle collapse toggle
  const handleToggleCollapse = useCallback(() => {
    const newCollapsed = !isCollapsed;
    setIsCollapsed(newCollapsed);
    onCollapse?.(newCollapsed);
  }, [isCollapsed, onCollapse]);

  // Calculate padding styles
  const paddingStyles = useMemo(() => {
    if (!padding) return {};
    
    if (typeof padding === 'number') {
      return { padding: `${padding}px` };
    }
    
    if (typeof padding === 'object') {
      return {
        paddingTop: padding.top ? `${padding.top}px` : undefined,
        paddingRight: padding.right ? `${padding.right}px` : undefined,
        paddingBottom: padding.bottom ? `${padding.bottom}px` : undefined,
        paddingLeft: padding.left ? `${padding.left}px` : undefined,
      };
    }
    
    return { padding: `${DEFAULT_PADDING}px` };
  }, [padding]);

  // Determine panel classes
  const panelClasses = [
    styles.panel,
    bordered && styles.panelBordered,
    elevated && styles.panelElevated,
    className,
  ].filter(Boolean).join(' ');

  const hasHeader = title || actions || collapsible;

  return (
    <div
      className={panelClasses}
      style={style}
      data-testid="panel"
    >
      {/* Header */}
      {hasHeader && (
        <div className={styles.panelHeader}>
          <div className={styles.panelHeaderContent}>
            {/* Collapse button and title */}
            {(title || collapsible) && (
              <div className={styles.panelHeaderTitle}>
                {collapsible && (
                  <button
                    onClick={handleToggleCollapse}
                    className={styles.panelCollapseButton}
                    aria-expanded={!isCollapsed}
                    aria-controls="panel-content"
                    aria-label={isCollapsed ? 'Expand panel' : 'Collapse panel'}
                  >
                    {isCollapsed ? (
                      <ChevronRightIcon className="w-4 h-4" />
                    ) : (
                      <ChevronDownIcon className="w-4 h-4" />
                    )}
                  </button>
                )}
                {title && (
                  <h3 className={styles.panelTitle}>{title}</h3>
                )}
              </div>
            )}

            {/* Description */}
            {description && !isCollapsed && (
              <p className={styles.panelDescription}>{description}</p>
            )}
          </div>

          {/* Actions */}
          {actions && !isCollapsed && (
            <div className={styles.panelActions}>
              {actions}
            </div>
          )}
        </div>
      )}

      {/* Content */}
      {!isCollapsed && (
        <div
          id="panel-content"
          className={styles.panelContent}
          style={paddingStyles}
        >
          {children}
        </div>
      )}
    </div>
  );
}