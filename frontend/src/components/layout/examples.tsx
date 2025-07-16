/**
 * Layout System Examples
 * ---
 * bwr-plots/frontend/src/components/layout/examples.tsx
 * ---
 * Complete examples showing how to use the layout system
 */

'use client';

import React from 'react';
import {
  ViewportProvider,
  AppShell,
  FlexLayout,
  FlexItem,
  ScrollArea,
  Panel,
  Grid,
  GridItem,
  useBreakpoint,
  useViewport,
} from './index';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 1: Three-Column Dashboard                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ThreeColumnDashboard() {
  return (
    <ViewportProvider>
      <AppShell>
        <FlexLayout orientation="horizontal" resizable>
          <FlexItem size="320px" minSize="280px" maxSize="400px">
            <Panel title="Data" bordered>
              <ScrollArea height="100%">
                <div className="p-4 space-y-4">
                  <div className="h-20 bg-gray-200 rounded"></div>
                  <div className="h-32 bg-gray-200 rounded"></div>
                  <div className="h-24 bg-gray-200 rounded"></div>
                  <div className="h-40 bg-gray-200 rounded"></div>
                </div>
              </ScrollArea>
            </Panel>
          </FlexItem>
          
          <FlexItem grow={1}>
            <Panel title="Visualization" bordered>
              <div className="h-full bg-gray-100 flex items-center justify-center">
                <p className="text-gray-600">Main Content Area</p>
              </div>
            </Panel>
          </FlexItem>
          
          <FlexItem size="384px" minSize="320px" maxSize="480px">
            <Panel title="Configuration" bordered collapsible>
              <ScrollArea height="100%">
                <div className="p-4 space-y-4">
                  <div className="h-16 bg-gray-200 rounded"></div>
                  <div className="h-20 bg-gray-200 rounded"></div>
                  <div className="h-24 bg-gray-200 rounded"></div>
                </div>
              </ScrollArea>
            </Panel>
          </FlexItem>
        </FlexLayout>
      </AppShell>
    </ViewportProvider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 2: Responsive Grid Dashboard                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ResponsiveGridDashboard() {
  return (
    <ViewportProvider>
      <AppShell 
        header={<DashboardHeader />}
        footer={<DashboardFooter />}
      >
        <Grid 
          columns={{ sm: 1, md: 2, lg: 3, xl: 4 }}
          gap={{ sm: 16, md: 20, lg: 24 }}
          className="p-6"
        >
          <GridItem colSpan={{ sm: 1, lg: 2 }}>
            <Panel title="Main Chart" bordered elevated>
              <div className="h-64 bg-blue-50 rounded-lg flex items-center justify-center">
                <p className="text-blue-600">Primary Visualization</p>
              </div>
            </Panel>
          </GridItem>
          
          <GridItem>
            <Panel title="Statistics" bordered>
              <div className="h-32 bg-green-50 rounded-lg flex items-center justify-center">
                <p className="text-green-600">Stats</p>
              </div>
            </Panel>
          </GridItem>
          
          <GridItem>
            <Panel title="Controls" bordered>
              <div className="h-32 bg-purple-50 rounded-lg flex items-center justify-center">
                <p className="text-purple-600">Controls</p>
              </div>
            </Panel>
          </GridItem>
          
          <GridItem colSpan={{ sm: 1, md: 2, xl: 1 }}>
            <Panel title="Data Table" bordered>
              <ScrollArea height="200px">
                <div className="p-4 space-y-2">
                  {Array.from({ length: 20 }, (_, i) => (
                    <div key={i} className="h-8 bg-gray-100 rounded"></div>
                  ))}
                </div>
              </ScrollArea>
            </Panel>
          </GridItem>
        </Grid>
      </AppShell>
    </ViewportProvider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 3: Vertical Layout with Sidebar                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function VerticalLayoutWithSidebar() {
  const { isMobile } = useViewport();
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(isMobile);

  React.useEffect(() => {
    setSidebarCollapsed(isMobile);
  }, [isMobile]);

  return (
    <ViewportProvider>
      <AppShell
        header={<NavigationHeader onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)} />}
        sidebar={<NavigationSidebar />}
        sidebarCollapsed={sidebarCollapsed}
        sidebarPosition="left"
      >
        <FlexLayout orientation="vertical" gap={16}>
          <FlexItem size="auto">
            <Panel title="Page Header" bordered>
              <div className="h-16 bg-indigo-50 rounded-lg flex items-center justify-center">
                <p className="text-indigo-600">Page Title & Actions</p>
              </div>
            </Panel>
          </FlexItem>
          
          <FlexItem grow={1}>
            <ScrollArea height="100%">
              <div className="p-6 space-y-6">
                <Panel title="Content Section 1" bordered>
                  <div className="h-64 bg-yellow-50 rounded-lg flex items-center justify-center">
                    <p className="text-yellow-600">Main Content</p>
                  </div>
                </Panel>
                
                <Panel title="Content Section 2" bordered collapsible>
                  <div className="h-48 bg-red-50 rounded-lg flex items-center justify-center">
                    <p className="text-red-600">Secondary Content</p>
                  </div>
                </Panel>
              </div>
            </ScrollArea>
          </FlexItem>
        </FlexLayout>
      </AppShell>
    </ViewportProvider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 4: Modal Layout                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ModalLayoutExample() {
  const [showModal, setShowModal] = React.useState(false);

  return (
    <ViewportProvider>
      <AppShell>
        <div className="p-8">
          <button
            onClick={() => setShowModal(true)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Open Modal
          </button>
        </div>
        
        {showModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="w-full max-w-2xl max-h-[80vh] mx-4">
              <Panel 
                title="Modal Example" 
                bordered 
                elevated
                actions={
                  <button
                    onClick={() => setShowModal(false)}
                    className="px-3 py-1 text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                }
              >
                <ScrollArea height="400px">
                  <div className="p-6 space-y-4">
                    <p>This is a modal built with the layout system.</p>
                    <div className="h-32 bg-gray-100 rounded"></div>
                    <div className="h-24 bg-gray-100 rounded"></div>
                    <div className="h-40 bg-gray-100 rounded"></div>
                  </div>
                </ScrollArea>
              </Panel>
            </div>
          </div>
        )}
      </AppShell>
    </ViewportProvider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 5: Responsive Breakpoint Demo                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ResponsiveBreakpointDemo() {
  const { breakpoint, width, height, isMobile, isTablet, isDesktop } = useViewport();
  const { isAbove, isBelow } = useBreakpoint();

  return (
    <ViewportProvider>
      <AppShell>
        <div className="p-8">
          <Panel title="Responsive Breakpoint Information" bordered>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold">Current Viewport:</h3>
                <p>Width: {width}px, Height: {height}px</p>
                <p>Breakpoint: {breakpoint}</p>
              </div>
              
              <div>
                <h3 className="font-semibold">Device Type:</h3>
                <p>Mobile: {isMobile ? 'Yes' : 'No'}</p>
                <p>Tablet: {isTablet ? 'Yes' : 'No'}</p>
                <p>Desktop: {isDesktop ? 'Yes' : 'No'}</p>
              </div>
              
              <div>
                <h3 className="font-semibold">Breakpoint Utilities:</h3>
                <p>Above lg: {isAbove('lg') ? 'Yes' : 'No'}</p>
                <p>Below xl: {isBelow('xl') ? 'Yes' : 'No'}</p>
              </div>
            </div>
          </Panel>
        </div>
      </AppShell>
    </ViewportProvider>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Supporting Components                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

function DashboardHeader() {
  return (
    <div className="h-16 px-6 flex items-center justify-between bg-white border-b">
      <h1 className="text-xl font-semibold">BWR Tools Dashboard</h1>
      <div className="flex items-center gap-4">
        <button className="px-3 py-1 text-sm bg-gray-100 rounded">Settings</button>
        <button className="px-3 py-1 text-sm bg-blue-500 text-white rounded">Export</button>
      </div>
    </div>
  );
}

function DashboardFooter() {
  return (
    <div className="h-12 px-6 flex items-center justify-center bg-gray-50 border-t">
      <p className="text-sm text-gray-600">© 2024 BWR Tools</p>
    </div>
  );
}

function NavigationHeader({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  return (
    <div className="h-16 px-6 flex items-center gap-4 bg-white border-b">
      <button
        onClick={onToggleSidebar}
        className="p-2 hover:bg-gray-100 rounded"
      >
        ☰
      </button>
      <h1 className="text-xl font-semibold">BWR Tools</h1>
    </div>
  );
}

function NavigationSidebar() {
  return (
    <div className="h-full bg-gray-900 text-white p-4">
      <nav className="space-y-2">
        <a href="#" className="block px-3 py-2 rounded hover:bg-gray-800">Dashboard</a>
        <a href="#" className="block px-3 py-2 rounded hover:bg-gray-800">Analytics</a>
        <a href="#" className="block px-3 py-2 rounded hover:bg-gray-800">Settings</a>
      </nav>
    </div>
  );
}