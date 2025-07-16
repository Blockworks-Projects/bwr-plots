/* ┌────────────────────────────────────────────────────────────────────────────────────┐
   │ Theme System                                                                        │
   │ Theme management and switching functionality                                        │
   └────────────────────────────────────────────────────────────────────────────────────┘ */

export type Theme = 'dark' | 'light' | 'system';

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THEME CONSTANTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const THEME_KEY = 'bwr-plots-theme';
const THEME_ATTRIBUTE = 'data-theme';
const DEFAULT_THEME: Theme = 'dark';

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SYSTEM THEME DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export function getSystemTheme(): 'dark' | 'light' {
  if (typeof window === 'undefined') return 'dark';
  
  return window.matchMedia('(prefers-color-scheme: dark)').matches 
    ? 'dark' 
    : 'light';
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THEME MANAGEMENT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export function getStoredTheme(): Theme {
  if (typeof window === 'undefined') return DEFAULT_THEME;
  
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === 'dark' || stored === 'light' || stored === 'system') {
      return stored;
    }
  } catch (error) {
    console.warn('Failed to get stored theme:', error);
  }
  
  return DEFAULT_THEME;
}

export function setStoredTheme(theme: Theme): void {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (error) {
    console.warn('Failed to store theme:', error);
  }
}

export function getResolvedTheme(theme: Theme): 'dark' | 'light' {
  return theme === 'system' ? getSystemTheme() : theme;
}

export function applyTheme(theme: Theme): void {
  if (typeof window === 'undefined') return;
  
  const resolvedTheme = getResolvedTheme(theme);
  document.documentElement.setAttribute(THEME_ATTRIBUTE, resolvedTheme);
  
  // Update meta theme-color for mobile browsers
  const metaThemeColor = document.querySelector('meta[name="theme-color"]');
  if (metaThemeColor) {
    metaThemeColor.setAttribute(
      'content',
      resolvedTheme === 'dark' ? '#09090B' : '#FAFAFA'
    );
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THEME INITIALIZATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export function initializeTheme(): void {
  const theme = getStoredTheme();
  applyTheme(theme);
  
  // Listen for system theme changes
  if (theme === 'system' && typeof window !== 'undefined') {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', () => {
        if (getStoredTheme() === 'system') {
          applyTheme('system');
        }
      });
    } else {
      // Legacy browsers
      mediaQuery.addListener(() => {
        if (getStoredTheme() === 'system') {
          applyTheme('system');
        }
      });
    }
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THEME SCRIPT FOR SSR
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// This script should be injected into the <head> to prevent flash of wrong theme
export const themeInitScript = `
(function() {
  const theme = localStorage.getItem('${THEME_KEY}') || '${DEFAULT_THEME}';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  document.documentElement.setAttribute('${THEME_ATTRIBUTE}', resolvedTheme);
})();
`;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THEME HOOK UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: 'dark' | 'light';
  setTheme: (theme: Theme) => void;
}

export function createThemeManager(): ThemeContextValue {
  const storedTheme = getStoredTheme();
  const resolvedTheme = getResolvedTheme(storedTheme);
  
  return {
    theme: storedTheme,
    resolvedTheme,
    setTheme: (newTheme: Theme) => {
      setStoredTheme(newTheme);
      applyTheme(newTheme);
    },
  };
}