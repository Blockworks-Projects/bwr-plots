/**
 * Feature Flag Management
 * ---
 * bwr-tools/frontend/src/lib/feature-flags.ts
 * ---
 * Centralized feature flag management with environment variable and localStorage support
 */

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface FeatureFlag {
  key: string;
  name: string;
  description: string;
  defaultValue: boolean;
  rolloutPercentage?: number;
  enabledFor?: string[]; // User IDs or email patterns
}

export interface FeatureFlagContext {
  flags: Record<string, boolean>;
  isEnabled: (flag: string) => boolean;
  enable: (flag: string) => void;
  disable: (flag: string) => void;
  toggle: (flag: string) => void;
  reset: () => void;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Feature Flag Definitions                                                           │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const FEATURE_FLAGS: Record<string, FeatureFlag> = {
  'layout-v2': {
    key: 'layout-v2',
    name: 'New Layout System',
    description: 'Enable the new flexible layout system with resizable panels',
    defaultValue: false,
    rolloutPercentage: 0, // Start with 0% rollout
  },
  'responsive-panels': {
    key: 'responsive-panels',
    name: 'Responsive Panels',
    description: 'Enable responsive behavior for panels on mobile devices',
    defaultValue: false,
    rolloutPercentage: 25,
  },
  'theme-switcher': {
    key: 'theme-switcher',
    name: 'Theme Switcher',
    description: 'Enable dark/light theme switching',
    defaultValue: true,
    rolloutPercentage: 100,
  },
  'performance-monitoring': {
    key: 'performance-monitoring',
    name: 'Performance Monitoring',
    description: 'Enable performance metrics collection',
    defaultValue: process.env.NODE_ENV === 'development',
    rolloutPercentage: 10,
  },
  'layout-debugger': {
    key: 'layout-debugger',
    name: 'Layout Debugger',
    description: 'Show layout debugging overlay',
    defaultValue: process.env.NODE_ENV === 'development',
  },
};

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Feature Flag Management Class                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

class FeatureFlagManager {
  private static instance: FeatureFlagManager;
  private flags: Map<string, boolean> = new Map();
  private localStorage: Storage | null = null;
  private readonly STORAGE_KEY = 'bwr-tools-feature-flags';

  private constructor() {
    this.initialize();
  }

  static getInstance(): FeatureFlagManager {
    if (!FeatureFlagManager.instance) {
      FeatureFlagManager.instance = new FeatureFlagManager();
    }
    return FeatureFlagManager.instance;
  }

  private initialize(): void {
    // Check if we're in a browser environment
    if (typeof window !== 'undefined') {
      this.localStorage = window.localStorage;
      this.loadFromStorage();
    }

    // Load from environment variables
    this.loadFromEnvironment();

    // Apply rollout percentages
    this.applyRolloutPercentages();
  }

  private loadFromStorage(): void {
    if (!this.localStorage) return;

    try {
      const stored = this.localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        Object.entries(parsed).forEach(([key, value]) => {
          this.flags.set(key, Boolean(value));
        });
      }
    } catch (error) {
      console.error('Failed to load feature flags from storage:', error);
    }
  }

  private loadFromEnvironment(): void {
    // Check for the new layout flag
    if (process.env.NEXT_PUBLIC_USE_NEW_LAYOUT === 'true') {
      this.flags.set('layout-v2', true);
    }

    // Parse feature flags from environment
    const envFlags = process.env.NEXT_PUBLIC_FEATURE_FLAGS?.split(',') || [];
    envFlags.forEach(flag => {
      const trimmed = flag.trim();
      if (trimmed && FEATURE_FLAGS[trimmed]) {
        this.flags.set(trimmed, true);
      }
    });
  }

  private applyRolloutPercentages(): void {
    Object.entries(FEATURE_FLAGS).forEach(([key, flag]) => {
      // Skip if already explicitly set
      if (this.flags.has(key)) return;

      // Apply rollout percentage
      if (flag.rolloutPercentage !== undefined) {
        const random = this.getUserBucket();
        const enabled = random < flag.rolloutPercentage;
        this.flags.set(key, enabled);
      } else {
        // Use default value
        this.flags.set(key, flag.defaultValue);
      }
    });
  }

  private getUserBucket(): number {
    // In a real app, this would be based on user ID for consistency
    // For now, use a random number or stored value
    if (this.localStorage) {
      let bucket = this.localStorage.getItem('bwr-tools-user-bucket');
      if (!bucket) {
        bucket = String(Math.random() * 100);
        this.localStorage.setItem('bwr-tools-user-bucket', bucket);
      }
      return parseFloat(bucket);
    }
    return Math.random() * 100;
  }

  private saveToStorage(): void {
    if (!this.localStorage) return;

    try {
      const toStore: Record<string, boolean> = {};
      this.flags.forEach((value, key) => {
        toStore[key] = value;
      });
      this.localStorage.setItem(this.STORAGE_KEY, JSON.stringify(toStore));
    } catch (error) {
      console.error('Failed to save feature flags to storage:', error);
    }
  }

  isEnabled(flag: string): boolean {
    return this.flags.get(flag) ?? FEATURE_FLAGS[flag]?.defaultValue ?? false;
  }

  enable(flag: string): void {
    this.flags.set(flag, true);
    this.saveToStorage();
  }

  disable(flag: string): void {
    this.flags.set(flag, false);
    this.saveToStorage();
  }

  toggle(flag: string): void {
    const current = this.isEnabled(flag);
    this.flags.set(flag, !current);
    this.saveToStorage();
  }

  reset(): void {
    this.flags.clear();
    if (this.localStorage) {
      this.localStorage.removeItem(this.STORAGE_KEY);
    }
    this.initialize();
  }

  getAllFlags(): Record<string, boolean> {
    const result: Record<string, boolean> = {};
    Object.keys(FEATURE_FLAGS).forEach(key => {
      result[key] = this.isEnabled(key);
    });
    return result;
  }
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Public API                                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

const manager = FeatureFlagManager.getInstance();

export const featureFlags = {
  isEnabled: (flag: string) => manager.isEnabled(flag),
  enable: (flag: string) => manager.enable(flag),
  disable: (flag: string) => manager.disable(flag),
  toggle: (flag: string) => manager.toggle(flag),
  reset: () => manager.reset(),
  getAllFlags: () => manager.getAllFlags(),
};

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ React Hook                                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

import { useState, useEffect } from 'react';

export function useFeatureFlag(flag: string): boolean {
  const [enabled, setEnabled] = useState(() => featureFlags.isEnabled(flag));

  useEffect(() => {
    // Check for updates periodically (useful for A/B testing)
    const interval = setInterval(() => {
      setEnabled(featureFlags.isEnabled(flag));
    }, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [flag]);

  return enabled;
}

export function useFeatureFlags(): FeatureFlagContext {
  const [flags, setFlags] = useState(() => featureFlags.getAllFlags());

  useEffect(() => {
    // Update flags when they change
    const interval = setInterval(() => {
      setFlags(featureFlags.getAllFlags());
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  return {
    flags,
    isEnabled: featureFlags.isEnabled,
    enable: (flag: string) => {
      featureFlags.enable(flag);
      setFlags(featureFlags.getAllFlags());
    },
    disable: (flag: string) => {
      featureFlags.disable(flag);
      setFlags(featureFlags.getAllFlags());
    },
    toggle: (flag: string) => {
      featureFlags.toggle(flag);
      setFlags(featureFlags.getAllFlags());
    },
    reset: () => {
      featureFlags.reset();
      setFlags(featureFlags.getAllFlags());
    },
  };
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Debug Component                                                                    │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function FeatureFlagDebugPanel() {
  const { flags, toggle, reset } = useFeatureFlags();

  if (process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <div className="fixed bottom-4 right-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-lg shadow-lg p-4 z-50 max-w-sm">
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
        Feature Flags
      </h3>
      <div className="space-y-2">
        {Object.entries(FEATURE_FLAGS).map(([key, flag]) => (
          <label
            key={key}
            className="flex items-center space-x-2 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={flags[key]}
              onChange={() => toggle(key)}
              className="rounded border-[var(--color-border)]"
            />
            <span className="text-xs text-[var(--color-text-secondary)]">
              {flag.name}
            </span>
          </label>
        ))}
      </div>
      <button
        onClick={reset}
        className="mt-3 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"
      >
        Reset to defaults
      </button>
    </div>
  );
}