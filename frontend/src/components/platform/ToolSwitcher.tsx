/**
 * Tool Switcher
 * ---
 * bwr-tools/frontend/src/components/platform/ToolSwitcher.tsx
 * ---
 * Navigation component for switching between tools
 */

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface Tool {
  id: string;
  name: string;
  description: string;
  icon: string;
  href: string;
  color: string;
  isActive: boolean;
}

interface ToolSwitcherProps {
  currentTool: string;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Available Tools Configuration                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

const AVAILABLE_TOOLS: Tool[] = [
  {
    id: 'plots',
    name: 'Plots',
    description: 'Create beautiful data visualizations',
    icon: '📊',
    href: '/plots',
    color: 'var(--color-primary)',
    isActive: true,
  },
  {
    id: 'data-explorer',
    name: 'Data Explorer',
    description: 'Explore and analyze your datasets',
    icon: '🔍',
    href: '/data-explorer',
    color: '#10B981', // Green
    isActive: false,
  },
  {
    id: 'ml-studio',
    name: 'ML Studio',
    description: 'Build and train machine learning models',
    icon: '🤖',
    href: '/ml-studio',
    color: '#F59E0B', // Amber
    isActive: false,
  },
  {
    id: 'api-builder',
    name: 'API Builder',
    description: 'Design and test REST APIs',
    icon: '🔌',
    href: '/api-builder',
    color: '#8B5CF6', // Purple
    isActive: false,
  },
];

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Tool Switcher Component                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ToolSwitcher({ currentTool }: ToolSwitcherProps) {
  const [showToolMenu, setShowToolMenu] = useState(false);

  const activeTool = AVAILABLE_TOOLS.find(tool => tool.id === currentTool) || AVAILABLE_TOOLS[0];

  return (
    <div className="relative">
      {/* Current Tool Button */}
      <Button
        variant="ghost"
        onClick={() => setShowToolMenu(!showToolMenu)}
        className="flex items-center space-x-2 px-3 py-2"
      >
        <span className="text-lg">{activeTool.icon}</span>
        <span className="font-medium text-[var(--color-text-primary)]">
          {activeTool.name}
        </span>
        <span className="text-xs text-[var(--color-text-muted)]">▼</span>
      </Button>

      {/* Tool Dropdown Menu */}
      {showToolMenu && (
        <>
          {/* Click outside to close */}
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setShowToolMenu(false)}
          />
          
          {/* Menu Content */}
          <div className="absolute top-full left-0 mt-2 w-80 bg-[var(--color-bg-elevated)] rounded-lg shadow-xl border border-[var(--color-border)] p-2 z-50">
            <div className="mb-2 px-3 py-2">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                Switch Tools
              </h3>
              <p className="text-xs text-[var(--color-text-muted)]">
                Select a tool to work with
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {AVAILABLE_TOOLS.map((tool) => (
                <Link
                  key={tool.id}
                  href={tool.href}
                  className={`
                    relative p-4 rounded-lg border transition-all
                    ${tool.isActive 
                      ? 'border-[var(--color-border)] hover:border-[var(--color-border-light)] hover:bg-[var(--color-hover-dark)]' 
                      : 'border-[var(--color-border)] opacity-60 cursor-not-allowed'
                    }
                    ${tool.id === currentTool ? 'bg-[var(--color-bg-tertiary)] border-[var(--color-primary)]' : ''}
                  `}
                  onClick={(e) => {
                    if (!tool.isActive) {
                      e.preventDefault();
                      return;
                    }
                    setShowToolMenu(false);
                  }}
                >
                  {/* Tool Icon */}
                  <div className="text-2xl mb-2">{tool.icon}</div>
                  
                  {/* Tool Info */}
                  <div>
                    <h4 className="text-sm font-medium text-[var(--color-text-primary)]">
                      {tool.name}
                    </h4>
                    <p className="text-xs text-[var(--color-text-muted)] mt-1">
                      {tool.description}
                    </p>
                  </div>

                  {/* Coming Soon Badge */}
                  {!tool.isActive && (
                    <div className="absolute top-2 right-2 px-2 py-1 bg-[var(--color-bg-secondary)] text-[var(--color-text-muted)] text-xs rounded">
                      Coming Soon
                    </div>
                  )}

                  {/* Active Indicator */}
                  {tool.id === currentTool && (
                    <div 
                      className="absolute bottom-0 left-0 right-0 h-1 rounded-b-lg"
                      style={{ backgroundColor: tool.color }}
                    />
                  )}
                </Link>
              ))}
            </div>

            {/* Quick Actions */}
            <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
              <Link
                href="/"
                className="flex items-center px-3 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-hover-dark)] rounded"
                onClick={() => setShowToolMenu(false)}
              >
                🏠 <span className="ml-2">Tools Dashboard</span>
              </Link>
              <Link
                href="/marketplace"
                className="flex items-center px-3 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-hover-dark)] rounded opacity-60 cursor-not-allowed"
                onClick={(e) => e.preventDefault()}
              >
                🛒 <span className="ml-2">Browse Marketplace</span>
                <span className="ml-auto text-xs text-[var(--color-text-muted)]">Soon</span>
              </Link>
            </div>
          </div>
        </>
      )}
    </div>
  );
}