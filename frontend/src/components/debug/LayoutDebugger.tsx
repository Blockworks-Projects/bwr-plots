/**
 * Layout Debugger Component
 * ---
 * bwr-plots/frontend/src/components/debug/LayoutDebugger.tsx
 * ---
 * Visual debugging tool for layout hierarchy
 */

'use client';

import React, { useEffect, useState } from 'react';

interface LayoutInfo {
  element: string;
  height: string;
  overflow: string;
  display: string;
  position: string;
  flexGrow?: string;
  flexShrink?: string;
  minHeight?: string;
  maxHeight?: string;
}

export function LayoutDebugger() {
  const [layoutInfo, setLayoutInfo] = useState<LayoutInfo[]>([]);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!isVisible) return;

    const analyzeLayout = () => {
      const info: LayoutInfo[] = [];
      
      // Key elements to analyze
      const selectors = [
        { selector: 'html', name: 'HTML' },
        { selector: 'body', name: 'Body' },
        { selector: '.h-screen', name: 'Tools Layout' },
        { selector: 'header', name: 'Platform Header' },
        { selector: 'main', name: 'Main Container' },
        { selector: '[class*="DashboardLayout"]', name: 'Dashboard Layout' },
        { selector: '[class*="w-80"]', name: 'Data Panel' },
        { selector: '[class*="flex-1"][class*="flex-col"]', name: 'Plot Canvas' },
        { selector: '[class*="w-96"]', name: 'Config Panel' },
      ];

      selectors.forEach(({ selector, name }) => {
        const el = document.querySelector(selector) as HTMLElement;
        if (el) {
          const computed = window.getComputedStyle(el);
          info.push({
            element: name,
            height: `${el.offsetHeight}px (${computed.height})`,
            overflow: computed.overflow || 'visible',
            display: computed.display,
            position: computed.position,
            flexGrow: computed.flexGrow,
            flexShrink: computed.flexShrink,
            minHeight: computed.minHeight,
            maxHeight: computed.maxHeight,
          });
        }
      });

      setLayoutInfo(info);
    };

    analyzeLayout();
    window.addEventListener('resize', analyzeLayout);
    return () => window.removeEventListener('resize', analyzeLayout);
  }, [isVisible]);

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="fixed bottom-4 right-4 z-50 bg-purple-600 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-purple-700 transition-colors"
        title="Toggle Layout Debugger"
      >
        🔍 Layout Debug
      </button>

      {/* Debug Panel */}
      {isVisible && (
        <div className="fixed top-20 right-4 z-50 bg-black/90 text-white p-4 rounded-lg shadow-2xl max-w-md max-h-[80vh] overflow-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-bold">Layout Debugger</h3>
            <button 
              onClick={() => setIsVisible(false)}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>

          <div className="space-y-3">
            {layoutInfo.map((info, index) => (
              <div key={index} className="border border-gray-700 rounded p-3">
                <h4 className="font-semibold text-purple-400 mb-2">{info.element}</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-400">Height:</span>
                    <span className="ml-2 font-mono">{info.height}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Overflow:</span>
                    <span className={`ml-2 font-mono ${
                      info.overflow === 'hidden' ? 'text-red-400' : 
                      info.overflow.includes('auto') ? 'text-green-400' : 
                      'text-gray-300'
                    }`}>
                      {info.overflow}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Display:</span>
                    <span className="ml-2 font-mono">{info.display}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Position:</span>
                    <span className="ml-2 font-mono">{info.position}</span>
                  </div>
                  {info.flexGrow && (
                    <div>
                      <span className="text-gray-400">Flex:</span>
                      <span className="ml-2 font-mono">
                        {info.flexGrow}/{info.flexShrink}
                      </span>
                    </div>
                  )}
                  {info.minHeight !== '0px' && (
                    <div>
                      <span className="text-gray-400">Min-H:</span>
                      <span className="ml-2 font-mono">{info.minHeight}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 p-3 bg-red-900/50 rounded">
            <h4 className="font-semibold text-red-400 mb-2">Issues Detected:</h4>
            <ul className="text-sm space-y-1">
              {layoutInfo.some(i => i.element === 'Main Container' && i.overflow === 'hidden') && (
                <li>• Main container has overflow-hidden preventing scrolling</li>
              )}
              {layoutInfo.filter(i => i.overflow.includes('auto')).length === 0 && (
                <li>• No scrollable containers found</li>
              )}
              {layoutInfo.some(i => parseInt(i.height) === 0) && (
                <li>• Some elements have zero height</li>
              )}
            </ul>
          </div>

          <div className="mt-4 text-xs text-gray-400">
            <p>Viewport: {window.innerWidth} × {window.innerHeight}px</p>
            <p>Screen: {window.screen.width} × {window.screen.height}px</p>
          </div>
        </div>
      )}
    </>
  );
}