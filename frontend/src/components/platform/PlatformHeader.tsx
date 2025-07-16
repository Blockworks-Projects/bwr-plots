/**
 * Platform Header
 * ---
 * bwr-tools/frontend/src/components/platform/PlatformHeader.tsx
 * ---
 * Persistent header with navigation and tool switching
 */

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { ToolSwitcher } from './ToolSwitcher';
import { ThemeToggle } from '@/contexts/ThemeContext';
import { useFeatureFlag } from '@/lib/feature-flags';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface PlatformHeaderProps {
  className?: string;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Platform Header Component                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function PlatformHeader({ className }: PlatformHeaderProps) {
  const pathname = usePathname();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const showThemeToggle = useFeatureFlag('theme-switcher');

  // Determine current tool from pathname
  const currentTool = pathname.split('/')[1] || 'home';

  return (
    <header className={`h-14 flex-shrink-0 bg-[var(--color-bg-secondary)] border-b border-[var(--color-border)] ${className || ''}`}>
      <div className="h-full px-4 flex items-center justify-between">
        {/* Left Section - Logo and Tool Switcher */}
        <div className="flex items-center space-x-4">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center space-x-2 text-[var(--color-text-primary)] hover:opacity-80 transition-opacity"
          >
            <div className="w-8 h-8 bg-[var(--color-primary)] rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">BW</span>
            </div>
            <span className="font-semibold text-lg hidden sm:block">BWR Tools</span>
          </Link>

          {/* Separator */}
          <div className="h-6 w-px bg-[var(--color-border)]" />

          {/* Tool Switcher */}
          <ToolSwitcher currentTool={currentTool} />
        </div>

        {/* Center Section - Global Search (Future) */}
        <div className="hidden lg:flex flex-1 max-w-xl mx-8">
          {/* Global search will go here */}
        </div>

        {/* Right Section - User Actions */}
        <div className="flex items-center space-x-3">
          {/* Theme Toggle */}
          {showThemeToggle && <ThemeToggle />}

          {/* Notifications (Future) */}
          <Button
            variant="ghost"
            size="sm"
            className="relative"
            title="Notifications"
          >
            🔔
            <span className="absolute top-1 right-1 w-2 h-2 bg-[var(--color-error)] rounded-full" />
          </Button>

          {/* Help */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => window.open('/docs', '_blank')}
            title="Help & Documentation"
          >
            ❓
          </Button>

          {/* User Menu */}
          <div className="relative">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-2"
            >
              <div className="w-7 h-7 bg-[var(--color-primary)] rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-medium">U</span>
              </div>
              <span className="hidden sm:block text-sm">User</span>
              <span className="text-xs">▼</span>
            </Button>

            {/* User Dropdown Menu */}
            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-[var(--color-bg-elevated)] rounded-lg shadow-lg border border-[var(--color-border)] py-2 z-50">
                <Link 
                  href="/settings/profile"
                  className="block px-4 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-hover-dark)] hover:text-[var(--color-text-primary)]"
                  onClick={() => setShowUserMenu(false)}
                >
                  Profile Settings
                </Link>
                <Link 
                  href="/settings/preferences"
                  className="block px-4 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-hover-dark)] hover:text-[var(--color-text-primary)]"
                  onClick={() => setShowUserMenu(false)}
                >
                  Preferences
                </Link>
                <div className="border-t border-[var(--color-border)] my-2" />
                <button
                  className="block w-full text-left px-4 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-hover-dark)] hover:text-[var(--color-text-primary)]"
                  onClick={() => {
                    setShowUserMenu(false);
                    // TODO: Implement logout
                    console.log('Logout');
                  }}
                >
                  Sign Out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Click outside to close menus */}
      {showUserMenu && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setShowUserMenu(false)}
        />
      )}
    </header>
  );
}