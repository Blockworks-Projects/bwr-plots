import { useMutation } from '@tanstack/react-query';
import { useState } from 'react';

interface UploadResponse {
  session_id: string;
  columns: Array<{ name: string; type: string }>;
  preview_data: Record<string, any>[];
  row_count: number;
  data_types: Record<string, string>;
}

export function useDataUpload() {
  const [uploadProgress, setUploadProgress] = useState(0);

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
    onSuccess: () => {
      setUploadProgress(0);
    },
    onError: () => {
      setUploadProgress(0);
    },
  });

  const uploadFile = (file: File) => {
    setUploadProgress(0);
    uploadMutation.mutate(file);
  };

  return {
    uploadFile,
    uploadProgress,
    isUploading: uploadMutation.isPending,
    uploadError: uploadMutation.error,
    uploadData: uploadMutation.data,
    isSuccess: uploadMutation.isSuccess,
    reset: uploadMutation.reset,
  };
} 