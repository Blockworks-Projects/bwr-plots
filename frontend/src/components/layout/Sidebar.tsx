'use client';

import { useState } from 'react';
import {
  ChartBarIcon,
  DocumentArrowUpIcon,
  EyeIcon,
  CogIcon,
  PresentationChartLineIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  QuestionMarkCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

interface SidebarProps {
  currentStep: number;
  onStepChange?: (step: number) => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function Sidebar({ 
  currentStep, 
  onStepChange, 
  isCollapsed = false, 
  onToggleCollapse 
}: SidebarProps) {
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);

  const steps = [
    {
      id: 1,
      name: 'Upload Data',
      description: 'Upload your CSV or Excel file',
      icon: DocumentArrowUpIcon,
      status: currentStep > 1 ? 'completed' : currentStep === 1 ? 'current' : 'upcoming',
    },
    {
      id: 2,
      name: 'Preview Data',
      description: 'Review your data structure',
      icon: EyeIcon,
      status: currentStep > 2 ? 'completed' : currentStep === 2 ? 'current' : 'upcoming',
    },
    {
      id: 3,
      name: 'Manipulate Data',
      description: 'Clean and transform your data',
      icon: CogIcon,
      status: currentStep > 3 ? 'completed' : currentStep === 3 ? 'current' : 'upcoming',
    },
    {
      id: 4,
      name: 'Configure Plot',
      description: 'Set up your visualization',
      icon: ChartBarIcon,
      status: currentStep > 4 ? 'completed' : currentStep === 4 ? 'current' : 'upcoming',
    },
    {
      id: 5,
      name: 'View Results',
      description: 'See and export your plot',
      icon: PresentationChartLineIcon,
      status: currentStep > 5 ? 'completed' : currentStep === 5 ? 'current' : 'upcoming',
    },
  ];

  const helpItems = [
    {
      id: 'help',
      name: 'Help & Documentation',
      icon: QuestionMarkCircleIcon,
      href: '#',
    },
    {
      id: 'troubleshoot',
      name: 'Troubleshooting',
      icon: ExclamationTriangleIcon,
      href: '#',
    },
  ];

  const getStatusStyles = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200 hover:bg-green-200';
      case 'current':
        return 'bg-blue-100 text-blue-800 border-blue-200 hover:bg-blue-200';
      case 'upcoming':
        return 'bg-gray-50 text-gray-500 border-gray-200 hover:bg-gray-100';
      default:
        return 'bg-gray-50 text-gray-500 border-gray-200';
    }
  };

  const getIconStyles = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600';
      case 'current':
        return 'text-blue-600';
      case 'upcoming':
        return 'text-gray-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div
      className={`bg-white border-r border-gray-200 transition-all duration-300 ${
        isCollapsed ? 'w-16' : 'w-64'
      } flex flex-col h-full`}
    >
      {/* Toggle button */}
      <div className="flex justify-end p-2 border-b border-gray-200">
        <button
          onClick={onToggleCollapse}
          className="p-1 rounded-md hover:bg-gray-100 text-gray-500 hover:text-gray-700"
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? (
            <ChevronRightIcon className="h-5 w-5" />
          ) : (
            <ChevronLeftIcon className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Steps Navigation */}
      <div className="flex-1 p-4 space-y-2">
        {!isCollapsed && (
          <h3 className="text-sm font-medium text-gray-900 mb-4">
            Workflow Steps
          </h3>
        )}
        
        <nav className="space-y-2">
          {steps.map((step) => {
            const Icon = step.icon;
            const isClickable = step.status === 'completed' || step.status === 'current';
            
            return (
              <button
                key={step.id}
                onClick={() => isClickable && onStepChange?.(step.id)}
                onMouseEnter={() => setHoveredStep(step.id)}
                onMouseLeave={() => setHoveredStep(null)}
                disabled={!isClickable}
                className={`w-full flex items-center p-3 rounded-lg border transition-all duration-200 ${
                  getStatusStyles(step.status)
                } ${isClickable ? 'cursor-pointer' : 'cursor-not-allowed'}`}
                title={isCollapsed ? step.name : undefined}
              >
                <div className="flex items-center">
                  <Icon className={`h-5 w-5 ${getIconStyles(step.status)} flex-shrink-0`} />
                  
                  {!isCollapsed && (
                    <div className="ml-3 text-left">
                      <div className="text-sm font-medium">{step.name}</div>
                      {(hoveredStep === step.id || step.status === 'current') && (
                        <div className="text-xs opacity-75 mt-1">
                          {step.description}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {!isCollapsed && step.status === 'completed' && (
                  <div className="ml-auto">
                    <div className="w-5 h-5 bg-green-600 rounded-full flex items-center justify-center">
                      <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                  </div>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Help Section */}
      <div className="border-t border-gray-200 p-4">
        {!isCollapsed && (
          <h3 className="text-sm font-medium text-gray-900 mb-3">
            Support
          </h3>
        )}
        
        <div className="space-y-2">
          {helpItems.map((item) => {
            const Icon = item.icon;
            
            return (
              <a
                key={item.id}
                href={item.href}
                className="flex items-center p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-50 transition-colors"
                title={isCollapsed ? item.name : undefined}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!isCollapsed && (
                  <span className="ml-3 text-sm">{item.name}</span>
                )}
              </a>
            );
          })}
        </div>
      </div>

      {/* Version info */}
      {!isCollapsed && (
        <div className="border-t border-gray-200 p-4">
          <div className="text-xs text-gray-500 text-center">
            BWR Plots v2.0.0
          </div>
        </div>
      )}
    </div>
  );
} 