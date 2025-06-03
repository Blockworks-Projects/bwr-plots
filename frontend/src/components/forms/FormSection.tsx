'use client';

import { ReactNode, useState } from 'react';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

interface FormSectionProps {
  title: string;
  description?: string;
  children: ReactNode;
  defaultCollapsed?: boolean;
  collapsible?: boolean;
  required?: boolean;
  className?: string;
  titleClassName?: string;
  contentClassName?: string;
  icon?: ReactNode;
  badge?: string;
  error?: boolean;
}

export function FormSection({
  title,
  description,
  children,
  defaultCollapsed = false,
  collapsible = false,
  required = false,
  className = '',
  titleClassName = '',
  contentClassName = '',
  icon,
  badge,
  error = false,
}: FormSectionProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  const toggleCollapsed = () => {
    if (collapsible) {
      setIsCollapsed(!isCollapsed);
    }
  };

  return (
    <div className={`bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Section Header */}
      <div
        className={`px-6 py-4 border-b border-gray-200 ${
          collapsible ? 'cursor-pointer hover:bg-gray-50' : ''
        } ${error ? 'border-red-200 bg-red-50' : ''}`}
        onClick={toggleCollapsed}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {icon && (
              <div className={`flex-shrink-0 ${error ? 'text-red-500' : 'text-gray-400'}`}>
                {icon}
              </div>
            )}
            
            <div>
              <h3 className={`text-lg font-medium leading-6 ${
                error ? 'text-red-900' : 'text-gray-900'
              } ${titleClassName}`}>
                {title}
                {required && <span className="text-red-500 ml-1">*</span>}
                {badge && (
                  <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {badge}
                  </span>
                )}
              </h3>
              
              {description && (
                <p className={`mt-1 text-sm ${
                  error ? 'text-red-700' : 'text-gray-500'
                }`}>
                  {description}
                </p>
              )}
            </div>
          </div>

          {collapsible && (
            <div className="flex-shrink-0">
              {isCollapsed ? (
                <ChevronRightIcon className={`h-5 w-5 ${
                  error ? 'text-red-400' : 'text-gray-400'
                }`} />
              ) : (
                <ChevronDownIcon className={`h-5 w-5 ${
                  error ? 'text-red-400' : 'text-gray-400'
                }`} />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Section Content */}
      {(!collapsible || !isCollapsed) && (
        <div className={`px-6 py-6 ${contentClassName}`}>
          {children}
        </div>
      )}
    </div>
  );
}

// Predefined section variants for common use cases
export function BasicFormSection({
  title,
  description,
  children,
  className = '',
}: {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <FormSection
      title={title}
      description={description}
      className={`${className}`}
    >
      {children}
    </FormSection>
  );
}

export function CollapsibleFormSection({
  title,
  description,
  children,
  defaultCollapsed = false,
  className = '',
}: {
  title: string;
  description?: string;
  children: ReactNode;
  defaultCollapsed?: boolean;
  className?: string;
}) {
  return (
    <FormSection
      title={title}
      description={description}
      collapsible
      defaultCollapsed={defaultCollapsed}
      className={className}
    >
      {children}
    </FormSection>
  );
}

export function AdvancedFormSection({
  title,
  description,
  children,
  className = '',
}: {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <FormSection
      title={title}
      description={description}
      collapsible
      defaultCollapsed
      badge="Advanced"
      className={className}
    >
      {children}
    </FormSection>
  );
}

// Form sections container for organizing multiple sections
export function FormSectionsContainer({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`space-y-6 ${className}`}>
      {children}
    </div>
  );
} 