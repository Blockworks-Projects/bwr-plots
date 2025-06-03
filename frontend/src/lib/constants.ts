// File size limits (in bytes)
export const FILE_SIZE_LIMITS = {
  VERCEL_MAX: 4.5 * 1024 * 1024, // 4.5MB - Vercel request limit
  GENERAL_MAX: 50 * 1024 * 1024, // 50MB - General file limit
} as const;

// Session configuration
export const SESSION_CONFIG = {
  TIMEOUT_MS: 30 * 60 * 1000, // 30 minutes
  STORAGE_KEY: 'bwr-plots-session',
} as const;

// API endpoints
export const API_ENDPOINTS = {
  DATA_UPLOAD: '/api/data/upload',
  DATA_PREVIEW: '/api/data/preview',
  DATA_MANIPULATE: '/api/data/manipulate',
  PLOTS_GENERATE: '/api/plots/generate',
  PLOTS_EXPORT: '/api/plots/export',
  PLOTS_TYPES: '/api/plots/types',
  HEALTH: '/api/health',
} as const;

// Supported file formats
export const SUPPORTED_FILE_TYPES = {
  CSV: { mime: 'text/csv', extension: '.csv' },
  XLSX: { mime: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', extension: '.xlsx' },
  XLS: { mime: 'application/vnd.ms-excel', extension: '.xls' },
} as const;

// Plot types
export const PLOT_TYPES = [
  'line',
  'scatter',
  'bar',
  'histogram',
  'box',
  'violin',
  'heatmap',
] as const;

// Query keys for React Query
export const QUERY_KEYS = {
  DATA_PREVIEW: 'dataPreview',
  PLOT_TYPES: 'plotTypes',
  PLOT_CONFIG: 'plotConfig',
} as const;

// UI constants
export const UI_CONFIG = {
  PAGINATION: {
    DEFAULT_PAGE_SIZE: 10,
    MAX_PAGE_SIZE: 100,
  },
  DEBOUNCE_MS: 300,
  TOAST_DURATION_MS: 5000,
} as const; 