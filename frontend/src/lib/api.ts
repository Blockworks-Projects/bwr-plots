import { 
  FileUploadResponse, 
  DataPreviewResponse, 
  PlotResponse, 
  PlotConfigResponse,
  ApiResponse 
} from '@/types/api';

// API client configuration
const API_BASE_URL = ''; // Always use relative URLs for Next.js API routes

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Generic API request function with error handling
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultOptions: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  // Remove Content-Type for FormData requests
  if (options.body instanceof FormData) {
    const headers = defaultOptions.headers as Record<string, string>;
    delete headers['Content-Type'];
  }

  try {
    const response = await fetch(url, defaultOptions);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.error || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData.code
      );
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    
    // Handle network errors, timeout, etc.
    throw new ApiError(
      error instanceof Error ? error.message : 'Network error occurred',
      0,
      'NETWORK_ERROR'
    );
  }
}

// API client functions
export const api = {
  // Data management endpoints
  data: {
    upload: async (file: File): Promise<FileUploadResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      
      return apiRequest<FileUploadResponse>('/api/data/upload', {
        method: 'POST',
        body: formData,
      });
    },
    
    preview: async (sessionId: string): Promise<DataPreviewResponse> => {
      return apiRequest<DataPreviewResponse>(`/api/data/preview/${sessionId}`);
    },
    
    manipulate: async (sessionId: string, operations: any[]): Promise<DataPreviewResponse> => {
      return apiRequest<DataPreviewResponse>('/api/data/manipulate', {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          operations,
        }),
      });
    },
  },
  
  // Plot generation endpoints
  plots: {
    generate: async (request: any): Promise<PlotResponse> => {
      return apiRequest<PlotResponse>('/api/plots/generate', {
        method: 'POST',
        body: JSON.stringify(request),
      });
    },
    
    generateFromFile: async (file: File, config: any): Promise<PlotResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('config', JSON.stringify(config));
      
      return apiRequest<PlotResponse>('/api/plots/generate-from-file', {
        method: 'POST',
        body: formData,
      });
    },
    
    validate: async (config: any): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> => {
      return apiRequest('/api/plots/validate', {
        method: 'POST',
        body: JSON.stringify(config),
      });
    },
    
    export: async (request: any): Promise<{ success: boolean; download_url?: string; error?: string }> => {
      return apiRequest('/api/plots/export', {
        method: 'POST',
        body: JSON.stringify(request),
      });
    },
    
    getTypes: async (): Promise<PlotConfigResponse> => {
      return apiRequest<PlotConfigResponse>('/api/plots/types');
    },
    
    getConfig: async (plotType?: string): Promise<any> => {
      const endpoint = plotType 
        ? `/api/plots/config?type=${plotType}`
        : '/api/plots/config';
      return apiRequest(endpoint);
    },
    
    getDataPreview: async (sessionId: string): Promise<any> => {
      return apiRequest(`/api/plots/data-preview?session_id=${sessionId}`);
    },
  },
  
  // Configuration endpoints
  config: {
    getWatermarks: async (): Promise<{ watermarks: string[] }> => {
      return apiRequest('/api/config/watermarks');
    },
    
    getPlotDefaults: async (plotType: string): Promise<any> => {
      return apiRequest(`/api/config/plot-defaults/${plotType}`);
    },
  },
  
  // Health check
  health: async (): Promise<{ status: string; timestamp: string }> => {
    return apiRequest('/api/health');
  },
};

// Error handling utilities
export function handleApiError(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.code) {
      case 'TIMEOUT':
        return 'The operation took too long to complete. Please try with a smaller file or simpler configuration.';
      case 'PAYLOAD_TOO_LARGE':
        return 'The file is too large. Please use a file smaller than 4.5MB.';
      case 'NETWORK_ERROR':
        return 'Network error. Please check your connection and try again.';
      default:
        return error.message;
    }
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'An unexpected error occurred';
}

// Retry utility for failed requests
export async function retryRequest<T>(
  requestFn: () => Promise<T>,
  maxRetries: number = 2,
  delay: number = 1000
): Promise<T> {
  let lastError: Error = new Error('Unknown error');
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');
      
      // Don't retry on client errors (4xx)
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
        throw error;
      }
      
      // Don't retry on the last attempt
      if (attempt === maxRetries) {
        break;
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, attempt)));
    }
  }
  
  throw lastError;
}

export { ApiError }; 