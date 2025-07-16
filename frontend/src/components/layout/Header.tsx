'use client';

import { useState } from 'react';
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline';

interface HeaderProps {
  currentStep?: number;
  totalSteps?: number;
}

export function Header({ currentStep = 1, totalSteps = 5 }: HeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navigation = [
    { name: 'Upload Data', href: '#', step: 1 },
    { name: 'Preview', href: '#', step: 2 },
    { name: 'Manipulate', href: '#', step: 3 },
    { name: 'Configure Plot', href: '#', step: 4 },
    { name: 'View Results', href: '#', step: 5 },
  ];

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between items-center">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-xl font-bold text-gray-900">
                BWR Plots
              </h1>
              <p className="text-xs text-gray-500">
                Data Visualization Tool
              </p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-8">
            {navigation.map((item) => (
              <button
                key={item.name}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  item.step === currentStep
                    ? 'bg-zinc-800 text-zinc-300'
                    : item.step < currentStep
                    ? 'text-green-600 hover:text-green-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                disabled={item.step > currentStep}
              >
                <span className="flex items-center">
                  <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-xs mr-2 ${
                    item.step === currentStep
                      ? 'bg-zinc-600 text-white'
                      : item.step < currentStep
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-300 text-gray-600'
                  }`}>
                    {item.step < currentStep ? '✓' : item.step}
                  </span>
                  {item.name}
                </span>
              </button>
            ))}
          </nav>

          {/* Progress indicator */}
          <div className="hidden md:flex items-center space-x-2">
            <span className="text-sm text-gray-500">
              Step {currentStep} of {totalSteps}
            </span>
            <div className="w-20 bg-gray-200 rounded-full h-2">
              <div
                className="bg-zinc-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(currentStep / totalSteps) * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className="bg-white rounded-md p-2 inline-flex items-center justify-center text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-zinc-500"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <span className="sr-only">Open main menu</span>
              {mobileMenuOpen ? (
                <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
              ) : (
                <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
              {navigation.map((item) => (
                <button
                  key={item.name}
                  className={`block px-3 py-2 rounded-md text-base font-medium w-full text-left transition-colors ${
                    item.step === currentStep
                      ? 'bg-zinc-800 text-zinc-300'
                      : item.step < currentStep
                      ? 'text-green-600 hover:text-green-700'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  disabled={item.step > currentStep}
                >
                  <span className="flex items-center">
                    <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-xs mr-3 ${
                      item.step === currentStep
                        ? 'bg-zinc-600 text-white'
                        : item.step < currentStep
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-300 text-gray-600'
                    }`}>
                      {item.step < currentStep ? '✓' : item.step}
                    </span>
                    {item.name}
                  </span>
                </button>
              ))}
              
              {/* Mobile progress indicator */}
              <div className="px-3 py-2">
                <div className="flex items-center justify-between text-sm text-gray-500 mb-2">
                  <span>Progress</span>
                  <span>{currentStep} of {totalSteps}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-zinc-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(currentStep / totalSteps) * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
} 