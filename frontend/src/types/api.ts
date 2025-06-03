// API Request Types
export interface FileUploadRequest {
  file: File;
}

export interface DataManipulationRequest {
  session_id: string;
  operations: DataOperation[];
}

export interface DataOperation {
  type: 'drop_columns' | 'rename_columns' | 'pivot';
  parameters: Record<string, any>;
}

export interface PlotRequest {
  session_id: string;
  plot_type: string;
  configuration: PlotConfiguration;
  data_processing?: DataProcessingConfig;
}

export interface PlotExportRequest {
  session_id: string;
  plot_data: any;
  format: 'html' | 'png' | 'svg' | 'pdf';
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface FileUploadResponse {
  session_id: string;
  columns: ColumnInfo[];
  preview_data: Record<string, any>[];
  row_count: number;
  data_types: Record<string, string>;
}

export interface DataPreviewResponse {
  columns: ColumnInfo[];
  preview_data: Record<string, any>[];
  row_count: number;
  data_types: Record<string, string>;
}

export interface PlotResponse {
  success: boolean;
  plot_json?: any;
  plot_html?: string;
  error?: string;
}

export interface PlotValidationResponse {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface PlotConfigResponse {
  plot_types: string[];
  plot_type_info?: PlotTypeInfo[];
  default_configurations: Record<string, any>;
}

export interface ColumnInfo {
  name: string;
  type: string;
  nullable?: boolean;
  unique_values?: number;
}

export interface PlotTypeInfo {
  id: string;
  name: string;
  description: string;
  required_columns: string[];
  optional_columns: string[];
  configuration_schema: any;
}

// Configuration Types
export interface PlotConfiguration {
  title?: string;
  subtitle?: string;
  source?: string;
  x_column?: string;
  y_column?: string;
  color_column?: string;
  size_column?: string;
  facet_column?: string;
  watermark?: string;
  prefix?: string;
  suffix?: string;
  axis_config?: AxisConfiguration;
  style_options?: PlotStyleOptions;
}

export interface AxisConfiguration {
  x_title?: string;
  y_title?: string;
  x_range?: [number, number];
  y_range?: [number, number];
  x_type?: 'linear' | 'log' | 'date' | 'category';
  y_type?: 'linear' | 'log' | 'date' | 'category';
}

export interface PlotStyleOptions {
  theme?: string;
  color_palette?: string[];
  font_family?: string;
  font_size?: number;
  background_color?: string;
  grid_visible?: boolean;
}

export interface DataProcessingConfig {
  date_column?: string;
  lookback_days?: number;
  date_window?: {
    start: string;
    end: string;
  };
  resampling?: {
    frequency: string;
    method: 'mean' | 'sum' | 'count' | 'first' | 'last';
  };
  smoothing?: {
    method: 'rolling' | 'exponential';
    window: number;
  };
  filtering?: {
    column: string;
    operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'in' | 'not_in';
    value: any;
  }[];
} 