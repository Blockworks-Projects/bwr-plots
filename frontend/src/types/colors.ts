/**
 * Color System
 * ---
 * bwr-plots/frontend/src/types/colors.ts
 * ---
 * Global color system following 60-30-10 principle
 * 60% neutral blacks/dark greys, 30% greys, 10% primary purple
 */

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Primary Colors (10% - Highlights & Accents)                                        │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const PRIMARY_COLORS = {
  // Base primary color
  primary: '#5B35D5',
  primaryLight: '#7C5DE8',
  primaryDark: '#4A2CB8',
  
  // Primary variations
  primaryAlpha: {
    5: 'rgba(91, 53, 213, 0.05)',
    10: 'rgba(91, 53, 213, 0.10)',
    20: 'rgba(91, 53, 213, 0.20)',
    30: 'rgba(91, 53, 213, 0.30)',
    50: 'rgba(91, 53, 213, 0.50)',
  },
  
  // Accent colors for highlights
  accent: '#6B46C1',
  accentLight: '#8B5CF6',
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Neutral Colors (60% - Main Content & Backgrounds)                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const NEUTRAL_COLORS = {
  // Pure blacks and dark greys
  black: '#000000',
  neutral900: '#0F0F0F',
  neutral800: '#1A1A1A',
  neutral700: '#262626',
  neutral600: '#333333',
  neutral500: '#404040',
  
  // Text colors
  textPrimary: '#FFFFFF',
  textSecondary: '#E5E5E5',
  textTertiary: '#CCCCCC',
  textMuted: '#999999',
  
  // Background colors
  bgPrimary: '#0F0F0F',
  bgSecondary: '#1A1A1A',
  bgTertiary: '#262626',
  bgCard: '#1F1F1F',
  bgElevated: '#2A2A2A',
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Grey Colors (30% - Supporting Elements)                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const GREY_COLORS = {
  // Light greys for borders, dividers, etc.
  grey100: '#F5F5F5',
  grey200: '#E5E5E5',
  grey300: '#D4D4D4',
  grey400: '#A3A3A3',
  grey500: '#737373',
  grey600: '#525252',
  grey700: '#404040',
  grey800: '#262626',
  grey900: '#171717',
  
  // Semantic grey colors
  border: '#333333',
  borderLight: '#404040',
  borderDark: '#262626',
  
  // Hover states
  hoverLight: '#F5F5F5',
  hoverDark: '#2A2A2A',
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Semantic Colors                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const SEMANTIC_COLORS = {
  // Status colors
  success: '#10B981',
  successLight: '#34D399',
  successDark: '#059669',
  
  error: '#EF4444',
  errorLight: '#F87171',
  errorDark: '#DC2626',
  
  warning: '#F59E0B',
  warningLight: '#FBBF24',
  warningDark: '#D97706',
  
  info: '#3B82F6',
  infoLight: '#60A5FA',
  infoDark: '#2563EB',
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Complete Color Palette                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const COLORS = {
  ...PRIMARY_COLORS,
  ...NEUTRAL_COLORS,
  ...GREY_COLORS,
  ...SEMANTIC_COLORS,
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ CSS Custom Properties                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const CSS_VARIABLES = {
  // Primary (10%)
  '--color-primary': PRIMARY_COLORS.primary,
  '--color-primary-light': PRIMARY_COLORS.primaryLight,
  '--color-primary-dark': PRIMARY_COLORS.primaryDark,
  '--color-accent': PRIMARY_COLORS.accent,
  
  // Neutral (60%)
  '--color-bg-primary': NEUTRAL_COLORS.bgPrimary,
  '--color-bg-secondary': NEUTRAL_COLORS.bgSecondary,
  '--color-bg-tertiary': NEUTRAL_COLORS.bgTertiary,
  '--color-bg-card': NEUTRAL_COLORS.bgCard,
  '--color-bg-elevated': NEUTRAL_COLORS.bgElevated,
  '--color-text-primary': NEUTRAL_COLORS.textPrimary,
  '--color-text-secondary': NEUTRAL_COLORS.textSecondary,
  '--color-text-tertiary': NEUTRAL_COLORS.textTertiary,
  '--color-text-muted': NEUTRAL_COLORS.textMuted,
  
  // Grey (30%)
  '--color-border': GREY_COLORS.border,
  '--color-border-light': GREY_COLORS.borderLight,
  '--color-border-dark': GREY_COLORS.borderDark,
  '--color-hover-light': GREY_COLORS.hoverLight,
  '--color-hover-dark': GREY_COLORS.hoverDark,
  
  // Semantic
  '--color-success': SEMANTIC_COLORS.success,
  '--color-error': SEMANTIC_COLORS.error,
  '--color-warning': SEMANTIC_COLORS.warning,
  '--color-info': SEMANTIC_COLORS.info,
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export type ColorToken = keyof typeof COLORS;
export type CSSVariable = keyof typeof CSS_VARIABLES;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utility Functions                                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Get a color value by token name
 */
export const getColor = (token: ColorToken): string => {
  const value = COLORS[token];
  // If it's a string, return it directly
  if (typeof value === 'string') {
    return value;
  }
  // If it's an object (like opacity variants), return a default or throw error
  throw new Error(`Color token "${token}" is not a direct color value. Use specific opacity variant instead.`);
};

/**
 * Generate CSS custom properties string
 */
export const generateCSSVariables = (): string => {
  return Object.entries(CSS_VARIABLES)
    .map(([key, value]) => `  ${key}: ${value};`)
    .join('\n');
};

/**
 * Color usage guidelines
 */
export const COLOR_USAGE = {
  primary: 'Use sparingly for CTAs, highlights, and key interactive elements (10%)',
  neutral: 'Main backgrounds, text, and content areas (60%)',
  grey: 'Borders, dividers, secondary UI elements (30%)',
  semantic: 'Status indicators, alerts, and feedback',
} as const;