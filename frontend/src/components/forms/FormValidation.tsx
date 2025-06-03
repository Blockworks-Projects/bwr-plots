'use client';

import { ReactNode, useState } from 'react';
import { 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  InformationCircleIcon,
  XCircleIcon 
} from '@heroicons/react/24/outline';

// Validation message types
type ValidationMessageType = 'error' | 'warning' | 'info' | 'success';

interface ValidationMessageProps {
  type: ValidationMessageType;
  message: string;
  details?: string[];
  className?: string;
}

interface FormValidationProps {
  errors?: Record<string, string | string[]>;
  warnings?: Record<string, string | string[]>;
  infos?: Record<string, string | string[]>;
  successes?: Record<string, string | string[]>;
  className?: string;
  showSummary?: boolean;
  title?: string;
}

// Individual validation message component
export function ValidationMessage({ 
  type, 
  message, 
  details = [], 
  className = '' 
}: ValidationMessageProps) {
  const getIcon = () => {
    switch (type) {
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-400" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400" />;
      case 'info':
        return <InformationCircleIcon className="h-5 w-5 text-blue-400" />;
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-400" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStyles = () => {
    switch (type) {
      case 'error':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'info':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'success':
        return 'bg-green-50 border-green-200 text-green-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  return (
    <div className={`rounded-md border p-4 ${getStyles()} ${className}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        <div className="ml-3">
          <p className="text-sm font-medium">
            {message}
          </p>
          {details.length > 0 && (
            <div className="mt-2">
              <ul className="list-disc list-inside text-sm space-y-1">
                {details.map((detail, index) => (
                  <li key={index}>{detail}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Inline field validation message
export function FieldValidationMessage({
  type,
  message,
  className = ''
}: {
  type: ValidationMessageType;
  message: string;
  className?: string;
}) {
  const getTextColor = () => {
    switch (type) {
      case 'error':
        return 'text-red-600';
      case 'warning':
        return 'text-yellow-600';
      case 'info':
        return 'text-blue-600';
      case 'success':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'error':
        return <XCircleIcon className="h-4 w-4" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-4 w-4" />;
      case 'info':
        return <InformationCircleIcon className="h-4 w-4" />;
      case 'success':
        return <CheckCircleIcon className="h-4 w-4" />;
      default:
        return null;
    }
  };

  return (
    <div className={`flex items-center mt-1 text-sm ${getTextColor()} ${className}`}>
      {getIcon() && (
        <div className="flex-shrink-0 mr-1">
          {getIcon()}
        </div>
      )}
      <span>{message}</span>
    </div>
  );
}

// Main form validation component
export function FormValidation({
  errors = {},
  warnings = {},
  infos = {},
  successes = {},
  className = '',
  showSummary = true,
  title
}: FormValidationProps) {
  const hasErrors = Object.keys(errors).length > 0;
  const hasWarnings = Object.keys(warnings).length > 0;
  const hasInfos = Object.keys(infos).length > 0;
  const hasSuccesses = Object.keys(successes).length > 0;
  
  const hasAnyMessages = hasErrors || hasWarnings || hasInfos || hasSuccesses;

  if (!hasAnyMessages || !showSummary) {
    return null;
  }

  const renderMessages = (
    messages: Record<string, string | string[]>, 
    type: ValidationMessageType
  ) => {
    return Object.entries(messages).map(([field, message]) => {
      const messageText = Array.isArray(message) ? message[0] : message;
      const details = Array.isArray(message) ? message.slice(1) : [];
      
      return (
        <ValidationMessage
          key={`${type}-${field}`}
          type={type}
          message={`${field}: ${messageText}`}
          details={details}
          className="mb-4 last:mb-0"
        />
      );
    });
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {title && (
        <h3 className="text-lg font-medium text-gray-900">
          {title}
        </h3>
      )}
      
      {/* Error messages - shown first */}
      {hasErrors && (
        <div className="space-y-3">
          {renderMessages(errors, 'error')}
        </div>
      )}
      
      {/* Warning messages */}
      {hasWarnings && (
        <div className="space-y-3">
          {renderMessages(warnings, 'warning')}
        </div>
      )}
      
      {/* Info messages */}
      {hasInfos && (
        <div className="space-y-3">
          {renderMessages(infos, 'info')}
        </div>
      )}
      
      {/* Success messages */}
      {hasSuccesses && (
        <div className="space-y-3">
          {renderMessages(successes, 'success')}
        </div>
      )}
    </div>
  );
}

// Form validation summary component
export function ValidationSummary({
  errorCount = 0,
  warningCount = 0,
  className = ''
}: {
  errorCount?: number;
  warningCount?: number;
  className?: string;
}) {
  if (errorCount === 0 && warningCount === 0) {
    return (
      <div className={`flex items-center text-sm text-green-600 ${className}`}>
        <CheckCircleIcon className="h-4 w-4 mr-1" />
        <span>All validations passed</span>
      </div>
    );
  }

  return (
    <div className={`text-sm ${className}`}>
      {errorCount > 0 && (
        <div className="flex items-center text-red-600 mb-1">
          <XCircleIcon className="h-4 w-4 mr-1" />
          <span>{errorCount} error{errorCount !== 1 ? 's' : ''} found</span>
        </div>
      )}
      {warningCount > 0 && (
        <div className="flex items-center text-yellow-600">
          <ExclamationTriangleIcon className="h-4 w-4 mr-1" />
          <span>{warningCount} warning{warningCount !== 1 ? 's' : ''} found</span>
        </div>
      )}
    </div>
  );
}

// HOC for adding validation to form components
export function withValidation<T extends object>(
  Component: React.ComponentType<T>
) {
  return function ValidatedComponent(
    props: T & { 
      validationErrors?: Record<string, string | string[]>;
      validationWarnings?: Record<string, string | string[]>;
    }
  ) {
    const { validationErrors, validationWarnings, ...restProps } = props;
    
    return (
      <div>
        <Component {...(restProps as T)} />
        {(validationErrors || validationWarnings) && (
          <FormValidation
            errors={validationErrors}
            warnings={validationWarnings}
            className="mt-4"
          />
        )}
      </div>
    );
  };
}

// Hook for managing form validation state
export function useFormValidation() {
  const [errors, setErrors] = useState<Record<string, string | string[]>>({});
  const [warnings, setWarnings] = useState<Record<string, string | string[]>>({});
  const [infos, setInfos] = useState<Record<string, string | string[]>>({});
  const [successes, setSuccesses] = useState<Record<string, string | string[]>>({});

  const addError = (field: string, message: string | string[]) => {
    setErrors(prev => ({ ...prev, [field]: message }));
  };

  const removeError = (field: string) => {
    setErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[field];
      return newErrors;
    });
  };

  const clearErrors = () => setErrors({});

  const addWarning = (field: string, message: string | string[]) => {
    setWarnings(prev => ({ ...prev, [field]: message }));
  };

  const removeWarning = (field: string) => {
    setWarnings(prev => {
      const newWarnings = { ...prev };
      delete newWarnings[field];
      return newWarnings;
    });
  };

  const clearWarnings = () => setWarnings({});

  const addInfo = (field: string, message: string | string[]) => {
    setInfos(prev => ({ ...prev, [field]: message }));
  };

  const addSuccess = (field: string, message: string | string[]) => {
    setSuccesses(prev => ({ ...prev, [field]: message }));
  };

  const clearAll = () => {
    setErrors({});
    setWarnings({});
    setInfos({});
    setSuccesses({});
  };

  const hasErrors = Object.keys(errors).length > 0;
  const hasWarnings = Object.keys(warnings).length > 0;

  return {
    errors,
    warnings,
    infos,
    successes,
    addError,
    removeError,
    clearErrors,
    addWarning,
    removeWarning,
    clearWarnings,
    addInfo,
    addSuccess,
    clearAll,
    hasErrors,
    hasWarnings,
    errorCount: Object.keys(errors).length,
    warningCount: Object.keys(warnings).length,
  };
}

 