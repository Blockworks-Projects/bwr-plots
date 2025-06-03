// Plot Types
export interface PlotConfig {
  type: PlotType;
  title?: string;
  subtitle?: string;
  source?: string;
  data_config: DataConfig;
  style_config: StyleConfig;
  axis_config: AxisConfig;
  export_config?: ExportConfig;
}

export type PlotType = 
  | 'line' 
  | 'bar' 
  | 'scatter' 
  | 'area' 
  | 'histogram' 
  | 'time_series'
  | 'horizontal_bar'
  | 'box'
  | 'heatmap';

export interface DataConfig {
  x_column: string;
  y_column?: string;
  color_column?: string;
  size_column?: string;
  facet_column?: string;
  text_column?: string;
  hover_columns?: string[];
  aggregation?: AggregationConfig;
}

export interface AggregationConfig {
  method: 'sum' | 'mean' | 'count' | 'min' | 'max' | 'median' | 'std';
  group_by?: string[];
}

export interface StyleConfig {
  theme: PlotTheme;
  color_palette: ColorPalette;
  font_family: string;
  font_size: number;
  background_color: string;
  grid_visible: boolean;
  legend_visible: boolean;
  legend_position: 'top' | 'bottom' | 'left' | 'right' | 'inside';
  watermark?: WatermarkConfig;
  annotations?: AnnotationConfig[];
}

export type PlotTheme = 
  | 'default' 
  | 'dark' 
  | 'minimal' 
  | 'presentation' 
  | 'publication' 
  | 'colorblind';

export type ColorPalette = 
  | 'default' 
  | 'viridis' 
  | 'plasma' 
  | 'inferno' 
  | 'magma' 
  | 'cividis' 
  | 'blues' 
  | 'reds' 
  | 'greens' 
  | 'oranges' 
  | 'purples' 
  | 'greys' 
  | 'rainbow' 
  | 'custom';

export interface WatermarkConfig {
  text: string;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
  opacity: number;
  font_size: number;
  color: string;
}

export interface AnnotationConfig {
  type: 'text' | 'arrow' | 'line' | 'rectangle' | 'circle';
  x: number;
  y: number;
  text?: string;
  color: string;
  size?: number;
}

export interface AxisConfig {
  x_title?: string;
  y_title?: string;
  x_range?: [number, number];
  y_range?: [number, number];
  x_type: AxisType;
  y_type: AxisType;
  x_format?: string;
  y_format?: string;
  x_tick_angle?: number;
  y_tick_angle?: number;
  x_grid_visible?: boolean;
  y_grid_visible?: boolean;
}

export type AxisType = 'linear' | 'log' | 'date' | 'category';

export interface ExportConfig {
  width: number;
  height: number;
  format: 'html' | 'png' | 'svg' | 'pdf' | 'json';
  dpi?: number;
  include_plotlyjs?: boolean | 'cdn' | 'inline';
}

// Plot Generation Types
export interface PlotJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  result?: PlotResult;
}

export interface PlotResult {
  plot_json: any;
  plot_html: string;
  metadata: PlotMetadata;
}

export interface PlotMetadata {
  data_points: number;
  processing_time: number;
  memory_usage: number;
  plot_size: {
    width: number;
    height: number;
  };
  data_hash: string;
}

// Plot Templates
export interface PlotTemplate {
  id: string;
  name: string;
  description: string;
  plot_type: PlotType;
  config: Partial<PlotConfig>;
  preview_image?: string;
  tags: string[];
  created_by: string;
  created_at: string;
}

export interface PlotTemplateCategory {
  id: string;
  name: string;
  description: string;
  templates: PlotTemplate[];
}

// Plot Validation
export interface PlotValidation {
  valid: boolean;
  errors: PlotValidationError[];
  warnings: PlotValidationWarning[];
  suggestions: PlotSuggestion[];
}

export interface PlotValidationError {
  type: 'missing_column' | 'invalid_column_type' | 'insufficient_data' | 'configuration_error';
  message: string;
  field?: string;
  suggested_fix?: string;
}

export interface PlotValidationWarning {
  type: 'performance' | 'visual' | 'data_quality';
  message: string;
  field?: string;
  impact: 'low' | 'medium' | 'high';
}

export interface PlotSuggestion {
  type: 'plot_type' | 'configuration' | 'data_processing';
  message: string;
  action?: string;
  confidence: number;
}

// Interactive Features
export interface PlotInteraction {
  type: 'hover' | 'click' | 'select' | 'zoom' | 'pan';
  enabled: boolean;
  config?: any;
}

export interface PlotAnimation {
  enabled: boolean;
  duration: number;
  easing: 'linear' | 'ease' | 'ease-in' | 'ease-out' | 'ease-in-out';
  frame_column?: string;
}

// Plot Sharing
export interface PlotShare {
  id: string;
  url: string;
  expires_at?: string;
  password_protected: boolean;
  download_enabled: boolean;
  view_count: number;
} 