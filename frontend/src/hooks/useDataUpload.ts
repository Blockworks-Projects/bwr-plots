import { useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { useSession } from '@/contexts/SessionContext';
import { DummyDataset, createCSVFile } from '@/lib/dummyData';

interface UploadResponse {
  session_id: string;
  columns: Array<{ name: string; type: string }>;
  preview_data: Record<string, any>[];
  row_count: number;
  total_rows: number;
  data_types: Record<string, string>;
  filename: string;
}

export function useDataUpload() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const { updateSession } = useSession();

  const uploadMutation = useMutation<UploadResponse, Error, File>({
    mutationFn: async (file: File) => {
      // Validate file size (4.5MB limit for Vercel)
      if (file.size > 4.5 * 1024 * 1024) {
        throw new Error('File size must be less than 4.5MB');
      }

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/data/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Upload failed');
      }

      return response.json();
    },
    onSuccess: async (data) => {
      setUploadProgress(0);
      
      // Fetch preview data
      try {
        const previewResponse = await fetch(`/api/data/preview/${data.session_id}?max_rows=100`);
        const previewData = await previewResponse.json();
        
        // Update session with the uploaded data and preview
        updateSession({
          uploadedData: {
            columns: data.columns.map(col => col.name),
            data: previewData.data || [],
            originalFileName: data.filename,
            rowCount: data.total_rows,
            dataTypes: data.data_types || {}
          },
          sessionId: data.session_id
        });
      } catch (error) {
        console.error('Failed to fetch preview data:', error);
        // Still update session even if preview fails
        updateSession({
          uploadedData: {
            columns: data.columns.map(col => col.name),
            data: [],
            originalFileName: data.filename,
            rowCount: data.total_rows,
            dataTypes: data.data_types || {}
          },
          sessionId: data.session_id
        });
      }
    },
    onError: () => {
      setUploadProgress(0);
    },
  });

  const uploadFile = (file: File) => {
    setUploadProgress(0);
    uploadMutation.mutate(file);
  };

  const uploadDummyData = (dataset: DummyDataset) => {
    // Convert dataset to CSV file and upload it like a real file
    const csvFile = createCSVFile(dataset);
    console.log(`[DUMMY_DATA] Created CSV file: ${csvFile.name} (${csvFile.size} bytes)`);
    console.log(`[DUMMY_DATA] Dataset type: ${dataset.type}, suggested for plot type`);
    
    // Upload the CSV file through the normal upload process
    uploadFile(csvFile);
    
    // Set the suggested plot type after a small delay to ensure upload completes
    setTimeout(() => {
      updateSession({
        suggestedPlotType: dataset.type
      });
    }, 100);
  };

  return {
    uploadFile,
    uploadDummyData,
    uploadProgress,
    isUploading: uploadMutation.isPending,
    error: uploadMutation.error?.message,
    uploadData: uploadMutation.data,
    isSuccess: uploadMutation.isSuccess,
    reset: uploadMutation.reset,
  };
} 