// Data Types
export interface DataSession {
  session_id: string;
  original_data: any[];
  current_data: any[];
  columns: ColumnMetadata[];
  row_count: number;
  created_at: string;
  last_modified: string;
}

export interface ColumnMetadata {
  name: string;
  type: DataType;
  nullable: boolean;
  unique_values: number;
  min_value?: number;
  max_value?: number;
  sample_values: any[];
  description?: string;
}

export type DataType = 
  | 'string' 
  | 'number' 
  | 'integer' 
  | 'float' 
  | 'boolean' 
  | 'date' 
  | 'datetime' 
  | 'categorical';

export interface DataPreview {
  data: Record<string, any>[];
  total_rows: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface DataValidation {
  valid: boolean;
  errors: DataValidationError[];
  warnings: DataValidationWarning[];
}

export interface DataValidationError {
  type: 'missing_column' | 'invalid_type' | 'empty_data' | 'duplicate_columns';
  message: string;
  column?: string;
  row?: number;
}

export interface DataValidationWarning {
  type: 'missing_values' | 'outliers' | 'inconsistent_format';
  message: string;
  column?: string;
  affected_rows: number;
}

// Data Manipulation Types
export interface ColumnDropOperation {
  type: 'drop_columns';
  parameters: {
    columns: string[];
  };
}

export interface ColumnRenameOperation {
  type: 'rename_columns';
  parameters: {
    mapping: Record<string, string>;
  };
}

export interface PivotOperation {
  type: 'pivot';
  parameters: {
    index_columns: string[];
    value_columns: string[];
    aggfunc: 'sum' | 'mean' | 'count' | 'min' | 'max' | 'first' | 'last';
  };
}

export type DataManipulationOperation = 
  | ColumnDropOperation 
  | ColumnRenameOperation 
  | PivotOperation;

// File Upload Types
export interface FileInfo {
  name: string;
  size: number;
  type: string;
  last_modified: number;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
  status: 'uploading' | 'processing' | 'complete' | 'error';
}

export interface FileValidation {
  valid: boolean;
  errors: string[];
  warnings: string[];
  file_info: FileInfo;
}

// Data Processing Types
export interface FilterConfig {
  column: string;
  operator: FilterOperator;
  value: any;
  enabled: boolean;
}

export type FilterOperator = 
  | 'equals' 
  | 'not_equals' 
  | 'greater_than' 
  | 'less_than' 
  | 'greater_equal' 
  | 'less_equal' 
  | 'contains' 
  | 'not_contains' 
  | 'starts_with' 
  | 'ends_with' 
  | 'in' 
  | 'not_in' 
  | 'is_null' 
  | 'is_not_null';

export interface SortConfig {
  column: string;
  direction: 'asc' | 'desc';
}

export interface GroupByConfig {
  columns: string[];
  aggregations: {
    column: string;
    function: 'sum' | 'mean' | 'count' | 'min' | 'max' | 'std' | 'var';
  }[];
}

// Data Export Types
export interface ExportConfig {
  format: 'csv' | 'xlsx' | 'json' | 'parquet';
  include_index: boolean;
  selected_columns?: string[];
  filters?: FilterConfig[];
  filename?: string;
} 