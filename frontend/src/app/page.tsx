'use client';

import { useState, useEffect } from 'react';
import { FileUpload } from '@/components/data/FileUpload';
import { DataPreview } from '@/components/data/DataPreview';
import { DataManipulation } from '@/components/data/DataManipulation';
import { PlotTypeSelector } from '@/components/plotting/PlotTypeSelector';
import { PlotConfiguration } from '@/components/plotting/PlotConfiguration';
import { PlotDisplay } from '@/components/plotting/PlotDisplay';
import { SessionDebugPanel } from '@/components/data/SessionDebugPanel';
import { useSession } from '@/hooks/useSession';
import { usePlotGeneration } from '@/hooks/usePlotGeneration';
import { PlotType } from '@/types/plots';
import { Button } from '@/components/ui/Button';

// Step indicator component
function StepIndicator({ currentStep }: { currentStep: number }) {
  const steps = [
    { number: 1, name: 'Upload Data', description: 'Upload your CSV or Excel file' },
    { number: 2, name: 'Preview Data', description: 'Review your data structure' },
    { number: 3, name: 'Manipulate Data', description: 'Modify columns and structure' },
    { number: 4, name: 'Configure Plot', description: 'Set up your visualization' },
    { number: 5, name: 'Generate Plot', description: 'Create and export your plot' },
  ];

  return (
    <div className="mb-8">
      <nav aria-label="Progress">
        <ol className="flex items-center justify-center space-x-4 md:space-x-8">
          {steps.map((step) => (
            <li key={step.number} className="flex items-center">
              <div className="flex flex-col items-center">
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                    currentStep > step.number
                      ? 'bg-blue-600 text-white'
                      : currentStep === step.number
                      ? 'bg-blue-100 text-blue-600 border-2 border-blue-600'
                      : 'bg-gray-100 text-gray-400'
                  }`}
                >
                  {currentStep > step.number ? (
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    step.number
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p className="text-sm font-medium text-gray-900">{step.name}</p>
                  <p className="text-xs text-gray-500 hidden md:block">{step.description}</p>
                </div>
              </div>
              {step.number < steps.length && (
                <div className={`hidden md:block w-16 h-px ml-4 ${
                  currentStep > step.number ? 'bg-blue-600' : 'bg-gray-200'
                }`} />
              )}
            </li>
          ))}
        </ol>
      </nav>
    </div>
  );
}

export default function Home() {
  const {
    sessionId,
    currentStep,
    setSessionId,
    nextStep,
    clearSession,
    hasActiveSession,
    hasStoredSession,
    restoreStoredSession
  } = useSession();

  // Plot generation state
  const [selectedPlotType, setSelectedPlotType] = useState<PlotType | null>(null);
  const [plotConfiguration, setPlotConfiguration] = useState<any>({});
  const [showRestoreDialog, setShowRestoreDialog] = useState(false);
  
  const {
    generatePlot,
    exportPlot,
    isGenerating,
    isExporting,
    error: plotError,
    lastGeneratedPlot,
    clearError,
    regenerate,
    debugInfo
  } = usePlotGeneration();

  // Check for stored session on mount
  useEffect(() => {
    if (hasStoredSession() && !hasActiveSession) {
      console.log('[Home] Found stored session, showing restore dialog');
      setShowRestoreDialog(true);
    }
  }, [hasStoredSession, hasActiveSession]);

  const handleUploadSuccess = (newSessionId: string) => {
    console.log(`[Home] Upload successful, setting session: ${newSessionId}`);
    setSessionId(newSessionId);
    nextStep(); // Move to step 2 (preview)
  };

  const handleRestoreSession = () => {
    console.log('[Home] User chose to restore session');
    if (restoreStoredSession()) {
      setShowRestoreDialog(false);
    }
  };

  const handleStartFresh = () => {
    console.log('[Home] User chose to start fresh');
    clearSession();
    setShowRestoreDialog(false);
  };

  const handleStartOver = () => {
    clearSession();
    setSelectedPlotType(null);
    setPlotConfiguration({});
    clearError();
  };

  const handlePlotTypeSelect = (plotType: PlotType) => {
    setSelectedPlotType(plotType);
    setPlotConfiguration({});
  };

  const handleGeneratePlot = async () => {
    if (!sessionId || !selectedPlotType) return;

    const result = await generatePlot({
      plotType: selectedPlotType,
      sessionId,
      configuration: plotConfiguration
    });

    if (result.success) {
      nextStep(); // Move to step 5 (display)
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Session Restore Dialog */}
      {showRestoreDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Previous Session Found
            </h3>
            <p className="text-gray-600 mb-6">
              We found a previous session with uploaded data. Would you like to continue where you left off or start fresh?
            </p>
            <div className="flex space-x-3">
              <Button onClick={handleRestoreSession} className="flex-1">
                Continue Previous Session
              </Button>
              <Button onClick={handleStartFresh} variant="outline" className="flex-1">
                Start Fresh
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">BWR Plots</h1>
              <p className="text-gray-600">Create beautiful data visualizations</p>
            </div>
            {hasActiveSession && (
              <button
                onClick={handleStartOver}
                className="text-sm text-gray-500 hover:text-gray-700 underline"
              >
                Start Over
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Step indicator */}
          <StepIndicator currentStep={currentStep} />

          {/* Step content */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  Upload Your Data File
                </h2>
                <p className="text-gray-600">
                  Start by uploading a CSV or Excel file containing your data
                </p>
              </div>
              <FileUpload onUploadSuccess={handleUploadSuccess} />
            </div>
          )}

          {currentStep === 2 && sessionId && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  Data Preview
                </h2>
                <p className="text-gray-600">
                  Review your uploaded data and column information
                </p>
              </div>
              <DataPreview
                sessionId={sessionId}
                onNext={nextStep}
              />
            </div>
          )}

          {currentStep === 3 && sessionId && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  Data Manipulation
                </h2>
                <p className="text-gray-600">
                  Modify your data structure by removing columns, renaming them, or pivoting
                </p>
              </div>
              <DataManipulation
                sessionId={sessionId}
                onNext={nextStep}
              />
            </div>
          )}

          {currentStep === 4 && sessionId && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  Plot Configuration
                </h2>
                <p className="text-gray-600">
                  Configure your plot type and visualization settings
                </p>
              </div>
              
              {/* Plot Type Selection */}
              {!selectedPlotType && (
                <PlotTypeSelector
                  selectedType={selectedPlotType}
                  onTypeSelect={handlePlotTypeSelect}
                />
              )}

              {/* Plot Configuration */}
              {selectedPlotType && (
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-medium text-gray-900">
                      Configuring {selectedPlotType.replace('_', ' ')} Plot
                    </h3>
                    <Button
                      variant="outline"
                      onClick={() => setSelectedPlotType(null)}
                    >
                      Change Plot Type
                    </Button>
                  </div>
                  
                  <PlotConfiguration
                    plotType={selectedPlotType}
                    sessionId={sessionId}
                    onConfigurationChange={setPlotConfiguration}
                    initialConfiguration={plotConfiguration}
                  />

                  {/* Error Display */}
                  {plotError && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <div className="flex items-start">
                        <div className="text-red-600 mr-2 mt-1">⚠️</div>
                        <div className="flex-1">
                          <div className="text-red-800 font-medium mb-2">Plot Generation Failed</div>
                          <div className="text-red-700 text-sm whitespace-pre-wrap">{plotError}</div>
                          {lastGeneratedPlot?.debugInfo && (
                            <details className="mt-3">
                              <summary className="text-red-600 cursor-pointer hover:text-red-800 text-sm">
                                Show debug details
                              </summary>
                              <div className="mt-2 bg-red-100 rounded p-2 text-xs">
                                <pre className="whitespace-pre-wrap overflow-x-auto">
                                  {JSON.stringify(lastGeneratedPlot.debugInfo, null, 2)}
                                </pre>
                              </div>
                            </details>
                          )}
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={clearError}
                          className="ml-2"
                        >
                          Dismiss
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Generate Button */}
                  <div className="flex justify-center">
                    <Button
                      onClick={handleGeneratePlot}
                      disabled={isGenerating || !plotConfiguration.x_column}
                      size="lg"
                      className="px-8"
                    >
                      {isGenerating ? 'Generating Plot...' : 'Generate Plot'}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {currentStep === 5 && sessionId && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  Plot Generation
                </h2>
                <p className="text-gray-600">
                  Your visualization is ready! Export or modify as needed
                </p>
              </div>
              
              <PlotDisplay
                plotData={lastGeneratedPlot?.plotData}
                plotHtml={lastGeneratedPlot?.plotHtml}
                isLoading={isGenerating}
                error={plotError}
                onExport={exportPlot}
                onRegenerateClick={regenerate}
              />

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  variant="outline"
                  onClick={() => {
                    setSelectedPlotType(null);
                    setPlotConfiguration({});
                    // Go back to step 4
                    const newStep = 4;
                    // This is a bit of a hack, but we need to manually set the step
                    const currentStepValue = currentStep;
                    while (currentStepValue > newStep) {
                      // We'd need to add a goToStep function to useSession
                      break;
                    }
                  }}
                >
                  Configure Different Plot
                </Button>
                
                <Button
                  onClick={regenerate}
                  disabled={isGenerating || !selectedPlotType}
                >
                  {isGenerating ? 'Regenerating...' : 'Regenerate Plot'}
                </Button>
                
                <Button
                  variant="outline"
                  onClick={handleStartOver}
                >
                  Start Over
                </Button>
              </div>
            </div>
          )}

          {/* No session state */}
          {!hasActiveSession && currentStep > 1 && (
            <div className="text-center space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <p className="text-yellow-800">
                  Your session has expired or no data is available. Please start by uploading a file.
                </p>
              </div>
              <button
                onClick={() => setSessionId(null)}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Start New Session
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Debug Panel */}
      <SessionDebugPanel
        sessionId={sessionId}
        currentStep={currentStep}
        hasActiveSession={hasActiveSession}
        hasStoredSession={hasStoredSession()}
        plotDebugInfo={debugInfo}
        plotError={plotError}
        plotConfiguration={plotConfiguration}
        selectedPlotType={selectedPlotType}
      />
    </div>
  );
}
