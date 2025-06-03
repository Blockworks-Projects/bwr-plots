import { z } from 'zod';

// Common validation rules
const emailSchema = z.string().email('Please enter a valid email address');
const urlSchema = z.string().url('Please enter a valid URL').optional().or(z.literal(''));
const positiveNumberSchema = z.number().positive('Must be a positive number');
const nonEmptyStringSchema = z.string().min(1, 'This field is required');

// File upload validation
export const fileUploadSchema = z.object({
  file: z.instanceof(File, { message: 'Please select a file' })
    .refine((file) => file.size <= 4.5 * 1024 * 1024, {
      message: 'File size must be less than 4.5MB'
    })
    .refine((file) => {
      const allowedTypes = [
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      ];
      return allowedTypes.includes(file.type);
    }, {
      message: 'File must be CSV or Excel format'
    }),
});

// Data manipulation validation
export const dataManipulationSchema = z.object({
  operation: z.enum(['drop_columns', 'rename_columns', 'pivot_data'], {
    errorMap: () => ({ message: 'Please select a valid operation' })
  }),
  columns: z.array(z.string()).min(1, 'Please select at least one column'),
  newNames: z.record(z.string()).optional(),
  pivotConfig: z.object({
    index: z.string().optional(),
    columns: z.string().optional(),
    values: z.string().optional(),
    aggFunc: z.enum(['mean', 'sum', 'count', 'min', 'max']).optional(),
  }).optional(),
});

// Plot configuration base schema
const plotConfigBaseSchema = z.object({
  title: z.string().min(1, 'Plot title is required'),
  subtitle: z.string().optional(),
  xAxisTitle: z.string().optional(),
  yAxisTitle: z.string().optional(),
  showLegend: z.boolean().default(true),
  showGrid: z.boolean().default(true),
  width: z.number().min(300).max(2000).default(800),
  height: z.number().min(200).max(1500).default(600),
});

// Plot data configuration
export const plotDataConfigSchema = z.object({
  xColumn: z.string().min(1, 'X-axis column is required'),
  yColumn: z.string().min(1, 'Y-axis column is required'),
  colorColumn: z.string().optional(),
  sizeColumn: z.string().optional(),
  groupColumn: z.string().optional(),
});

// Specific plot type schemas
export const linePlayConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('line'),
  data: plotDataConfigSchema,
  lineStyle: z.enum(['solid', 'dashed', 'dotted']).default('solid'),
  lineWidth: z.number().min(1).max(10).default(2),
  showMarkers: z.boolean().default(true),
  markerSize: z.number().min(2).max(20).default(6),
  smoothing: z.number().min(0).max(1).default(0),
});

export const barPlotConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('bar'),
  data: plotDataConfigSchema,
  orientation: z.enum(['vertical', 'horizontal']).default('vertical'),
  barWidth: z.number().min(0.1).max(1).default(0.8),
  showValues: z.boolean().default(false),
});

export const scatterPlotConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('scatter'),
  data: plotDataConfigSchema,
  markerSize: z.number().min(2).max(50).default(8),
  opacity: z.number().min(0.1).max(1).default(0.7),
  showTrendline: z.boolean().default(false),
});

export const heatmapConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('heatmap'),
  data: z.object({
    xColumn: z.string().min(1, 'X-axis column is required'),
    yColumn: z.string().min(1, 'Y-axis column is required'),
    valueColumn: z.string().min(1, 'Value column is required'),
  }),
  colorScale: z.enum(['viridis', 'plasma', 'blues', 'reds', 'greens']).default('viridis'),
  showColorbar: z.boolean().default(true),
});

export const histogramConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('histogram'),
  data: z.object({
    column: z.string().min(1, 'Column is required'),
    groupColumn: z.string().optional(),
  }),
  bins: z.number().min(5).max(100).default(20),
  showDensity: z.boolean().default(false),
  opacity: z.number().min(0.1).max(1).default(0.7),
});

export const boxPlotConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('box'),
  data: z.object({
    yColumn: z.string().min(1, 'Y-axis column is required'),
    groupColumn: z.string().optional(),
  }),
  showPoints: z.enum(['all', 'outliers', 'none']).default('outliers'),
  notched: z.boolean().default(false),
});

export const violinPlotConfigSchema = plotConfigBaseSchema.extend({
  plotType: z.literal('violin'),
  data: z.object({
    yColumn: z.string().min(1, 'Y-axis column is required'),
    groupColumn: z.string().optional(),
  }),
  showBox: z.boolean().default(true),
  showPoints: z.boolean().default(false),
});

// Union of all plot configuration schemas
export const plotConfigSchema = z.discriminatedUnion('plotType', [
  linePlayConfigSchema,
  barPlotConfigSchema,
  scatterPlotConfigSchema,
  heatmapConfigSchema,
  histogramConfigSchema,
  boxPlotConfigSchema,
  violinPlotConfigSchema,
]);

// Data processing configuration
export const dataProcessingSchema = z.object({
  filterConfig: z.object({
    enabled: z.boolean().default(false),
    column: z.string().optional(),
    operation: z.enum(['>', '<', '>=', '<=', '==', '!=']).optional(),
    value: z.union([z.string(), z.number()]).optional(),
    dateRange: z.object({
      start: z.date().optional(),
      end: z.date().optional(),
    }).optional(),
  }).optional(),
  
  resamplingConfig: z.object({
    enabled: z.boolean().default(false),
    frequency: z.enum(['D', 'H', 'M', 'Q', 'Y']).optional(),
    method: z.enum(['mean', 'sum', 'min', 'max', 'first', 'last']).optional(),
    dateColumn: z.string().optional(),
  }).optional(),
  
  smoothingConfig: z.object({
    enabled: z.boolean().default(false),
    method: z.enum(['rolling_mean', 'exponential', 'lowess']).optional(),
    window: z.number().min(2).max(100).optional(),
    alpha: z.number().min(0.01).max(1).optional(),
  }).optional(),
});

// Export configuration
export const exportConfigSchema = z.object({
  format: z.enum(['html', 'png', 'svg', 'pdf']),
  filename: z.string().min(1, 'Filename is required'),
  width: z.number().min(300).max(3000).default(1200),
  height: z.number().min(200).max(2000).default(800),
  includeData: z.boolean().default(false),
  watermark: z.string().optional(),
});

// Settings and preferences
export const userPreferencesSchema = z.object({
  theme: z.enum(['light', 'dark', 'auto']).default('light'),
  defaultPlotType: z.enum(['line', 'bar', 'scatter', 'heatmap', 'histogram', 'box', 'violin']).default('line'),
  autoSave: z.boolean().default(true),
  showTooltips: z.boolean().default(true),
  defaultColors: z.array(z.string()).default(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
});

// Session configuration
export const sessionConfigSchema = z.object({
  sessionId: z.string().min(1, 'Session ID is required'),
  expiresAt: z.date(),
  metadata: z.record(z.unknown()).optional(),
});

// API request/response validation schemas
export const apiResponseSchema = z.object({
  success: z.boolean(),
  data: z.unknown().optional(),
  error: z.string().optional(),
  message: z.string().optional(),
});

export const dataPreviewResponseSchema = z.object({
  columns: z.array(z.object({
    name: z.string(),
    type: z.string(),
    nullable: z.boolean().optional(),
  })),
  preview: z.array(z.record(z.unknown())),
  rowCount: z.number(),
  summary: z.record(z.unknown()).optional(),
});

export const plotGenerationRequestSchema = z.object({
  sessionId: z.string().min(1, 'Session ID is required'),
  plotConfig: plotConfigSchema,
  dataProcessing: dataProcessingSchema.optional(),
  exportConfig: exportConfigSchema.optional(),
});

// Validation helper functions
export function validateFileSize(file: File, maxSizeMB: number = 4.5): boolean {
  return file.size <= maxSizeMB * 1024 * 1024;
}

export function validateFileType(file: File, allowedTypes: string[]): boolean {
  return allowedTypes.includes(file.type);
}

export function validateRequired<T>(value: T | null | undefined, fieldName: string): T {
  if (value === null || value === undefined || value === '') {
    throw new Error(`${fieldName} is required`);
  }
  return value;
}

export function validateRange(value: number, min: number, max: number, fieldName: string): boolean {
  if (value < min || value > max) {
    throw new Error(`${fieldName} must be between ${min} and ${max}`);
  }
  return true;
}

// Type exports for use in components
export type FileUploadData = z.infer<typeof fileUploadSchema>;
export type DataManipulationData = z.infer<typeof dataManipulationSchema>;
export type PlotConfigData = z.infer<typeof plotConfigSchema>;
export type DataProcessingData = z.infer<typeof dataProcessingSchema>;
export type ExportConfigData = z.infer<typeof exportConfigSchema>;
export type UserPreferencesData = z.infer<typeof userPreferencesSchema>;
export type SessionConfigData = z.infer<typeof sessionConfigSchema>;
export type DataPreviewResponseData = z.infer<typeof dataPreviewResponseSchema>;
export type PlotGenerationRequestData = z.infer<typeof plotGenerationRequestSchema>; 