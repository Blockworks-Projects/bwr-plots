# BWR Plots Implementation Guides

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Creating Layout Components](#creating-layout-components)
3. [Migrating Existing Components](#migrating-existing-components)
4. [Testing Procedures](#testing-procedures)
5. [Performance Optimization](#performance-optimization)
6. [Debugging Tools](#debugging-tools)

---

## Development Environment Setup

### Prerequisites
```bash
# Required versions
node >= 18.0.0
npm >= 9.0.0
git >= 2.30.0

# Optional but recommended
pnpm >= 8.0.0  # Faster alternative to npm
```

### Initial Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-org/bwr-plots.git
cd bwr-plots/frontend

# 2. Install dependencies
npm install

# 3. Set up environment variables
cp .env.example .env.local

# 4. Configure environment
cat >> .env.local << EOL
# Feature flags
NEXT_PUBLIC_USE_NEW_LAYOUT=true
NEXT_PUBLIC_ENABLE_LAYOUT_DEBUGGER=true
NEXT_PUBLIC_ENABLE_PERFORMANCE_MONITORING=true

# Development tools
NEXT_PUBLIC_SHOW_RENDER_METRICS=true
NEXT_PUBLIC_HIGHLIGHT_LAYOUT_UPDATES=true
EOL

# 5. Install development tools
npm install -D @welldone-software/why-did-you-render
npm install -D @tanstack/react-query-devtools
npm install -D web-vitals

# 6. Set up git hooks
npm run prepare
```

### VS Code Configuration
```json
// .vscode/settings.json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  "tailwindCSS.experimental.classRegex": [
    ["cn\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"],
    ["cva\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"]
  ],
  "files.associations": {
    "*.css": "tailwindcss"
  }
}
```

### Development Scripts
```bash
# Start development server with all features
npm run dev:full

# Start with layout debugger
npm run dev:debug

# Start with performance monitoring
npm run dev:perf

# Run type checking in watch mode
npm run type-check:watch

# Run tests in watch mode
npm run test:watch
```

### Browser Extensions
- React Developer Tools
- Redux DevTools (if using Redux)
- Lighthouse
- axe DevTools (accessibility)
- CSS Grid Inspector

---

## Creating Layout Components

### Component Template
```typescript
// components/layout/MyLayout/MyLayout.tsx
import { forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { BaseLayoutProps, ResponsiveValue } from '@/types/layout';
import { useResponsive } from '@/hooks/useResponsive';
import styles from './MyLayout.module.css';

export interface MyLayoutProps extends BaseLayoutProps {
  /**
   * The layout variant to use
   * @default 'default'
   */
  variant?: 'default' | 'compact' | 'spacious';
  
  /**
   * Spacing between child elements
   * @default 'md'
   */
  spacing?: ResponsiveValue<'xs' | 'sm' | 'md' | 'lg' | 'xl'>;
  
  /**
   * Whether to add dividers between items
   * @default false
   */
  dividers?: boolean;
}

/**
 * MyLayout component for organizing content
 * 
 * @example
 * ```tsx
 * <MyLayout spacing={{ base: 'sm', md: 'lg' }} dividers>
 *   <div>Content 1</div>
 *   <div>Content 2</div>
 * </MyLayout>
 * ```
 */
export const MyLayout = forwardRef<HTMLDivElement, MyLayoutProps>(
  (
    {
      variant = 'default',
      spacing = 'md',
      dividers = false,
      children,
      className,
      ...props
    },
    ref
  ) => {
    const responsive = useResponsive();
    const currentSpacing = responsive.getValue(spacing);
    
    return (
      <div
        ref={ref}
        className={cn(
          // Base styles
          styles.root,
          
          // Variant styles
          styles[`variant-${variant}`],
          
          // Spacing styles
          styles[`spacing-${currentSpacing}`],
          
          // Conditional styles
          dividers && styles.dividers,
          
          // Custom className
          className
        )}
        data-spacing={currentSpacing}
        {...props}
      >
        {children}
      </div>
    );
  }
);

MyLayout.displayName = 'MyLayout';
```

### Component Styles
```css
/* components/layout/MyLayout/MyLayout.module.css */
.root {
  display: flex;
  flex-direction: column;
}

/* Variants */
.variant-default {
  /* Default styles */
}

.variant-compact {
  /* Compact variant styles */
}

.variant-spacious {
  /* Spacious variant styles */
}

/* Spacing */
.spacing-xs { gap: var(--spacing-xs); }
.spacing-sm { gap: var(--spacing-sm); }
.spacing-md { gap: var(--spacing-md); }
.spacing-lg { gap: var(--spacing-lg); }
.spacing-xl { gap: var(--spacing-xl); }

/* Dividers */
.dividers > *:not(:last-child) {
  border-bottom: 1px solid var(--border-color);
  padding-bottom: var(--current-spacing);
}
```

### Component Tests
```typescript
// components/layout/MyLayout/MyLayout.test.tsx
import { render, screen } from '@testing-library/react';
import { MyLayout } from './MyLayout';

describe('MyLayout', () => {
  it('renders children correctly', () => {
    render(
      <MyLayout>
        <div>Child 1</div>
        <div>Child 2</div>
      </MyLayout>
    );
    
    expect(screen.getByText('Child 1')).toBeInTheDocument();
    expect(screen.getByText('Child 2')).toBeInTheDocument();
  });

  it('applies variant styles', () => {
    const { container } = render(
      <MyLayout variant="compact">Content</MyLayout>
    );
    
    expect(container.firstChild).toHaveClass('variant-compact');
  });

  it('handles responsive spacing', () => {
    const { container } = render(
      <MyLayout spacing={{ base: 'sm', md: 'lg' }}>
        Content
      </MyLayout>
    );
    
    // Test will depend on current viewport
    const element = container.firstChild as HTMLElement;
    expect(element.dataset.spacing).toMatch(/^(sm|lg)$/);
  });
});
```

### Component Story
```typescript
// components/layout/MyLayout/MyLayout.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { MyLayout } from './MyLayout';

const meta: Meta<typeof MyLayout> = {
  title: 'Layout/MyLayout',
  component: MyLayout,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: (
      <>
        <div className="p-4 bg-gray-100">Item 1</div>
        <div className="p-4 bg-gray-100">Item 2</div>
        <div className="p-4 bg-gray-100">Item 3</div>
      </>
    ),
  },
};

export const WithDividers: Story = {
  args: {
    ...Default.args,
    dividers: true,
    spacing: 'lg',
  },
};

export const ResponsiveSpacing: Story = {
  args: {
    ...Default.args,
    spacing: { base: 'sm', md: 'md', lg: 'lg' },
  },
};
```

### Export and Index
```typescript
// components/layout/MyLayout/index.ts
export { MyLayout } from './MyLayout';
export type { MyLayoutProps } from './MyLayout';

// components/layout/index.ts
export * from './Stack';
export * from './Grid';
export * from './Container';
export * from './MyLayout'; // Add new export
```

---

## Migrating Existing Components

### Migration Checklist
- [ ] Analyze current component structure
- [ ] Identify layout responsibilities
- [ ] Create feature flag for gradual rollout
- [ ] Implement new version using layout primitives
- [ ] Add comprehensive tests
- [ ] Set up A/B testing
- [ ] Document breaking changes
- [ ] Plan deprecation timeline

### Step-by-Step Migration

#### 1. Analyze Current Component
```typescript
// OLD: components/PlotDisplay.tsx
export function PlotDisplay({ data, config }) {
  return (
    <div className="plot-display">
      <div className="plot-header">
        <h2 className="plot-title">{config.title}</h2>
        <div className="plot-actions">
          <button>Export</button>
          <button>Share</button>
        </div>
      </div>
      <div className="plot-body">
        <div className="plot-container">
          <canvas ref={canvasRef} />
        </div>
        <div className="plot-legend">
          {/* Legend items */}
        </div>
      </div>
      <div className="plot-footer">
        <p className="plot-description">{config.description}</p>
      </div>
    </div>
  );
}
```

#### 2. Create Feature Flag
```typescript
// lib/feature-flags.ts
export const useFeatureFlag = (flag: string) => {
  const flags = {
    'new-plot-display': process.env.NEXT_PUBLIC_NEW_PLOT_DISPLAY === 'true',
    // ... other flags
  };
  
  return flags[flag] ?? false;
};
```

#### 3. Implement New Version
```typescript
// NEW: components/PlotDisplay.new.tsx
import { Stack, Grid, Container } from '@/components/layout';
import { Card, CardHeader, CardContent, CardFooter } from '@/components/ui';

export function PlotDisplayNew({ data, config }) {
  return (
    <Card>
      <CardHeader>
        <Grid cols={2} align="center">
          <Heading level={2}>{config.title}</Heading>
          <Stack direction="horizontal" spacing="sm" justify="end">
            <Button variant="secondary" size="sm">Export</Button>
            <Button variant="secondary" size="sm">Share</Button>
          </Stack>
        </Grid>
      </CardHeader>
      
      <CardContent>
        <Grid cols={{ base: 1, lg: '3fr 1fr' }} gap="lg">
          <AspectRatio ratio={16/9}>
            <canvas ref={canvasRef} className="w-full h-full" />
          </AspectRatio>
          
          <Stack spacing="sm">
            <Heading level={3} size="sm">Legend</Heading>
            {/* Legend items using Stack */}
          </Stack>
        </Grid>
      </CardContent>
      
      {config.description && (
        <CardFooter>
          <Text size="sm" color="muted">
            {config.description}
          </Text>
        </CardFooter>
      )}
    </Card>
  );
}
```

#### 4. Create Compatibility Wrapper
```typescript
// components/PlotDisplay.tsx
import { useFeatureFlag } from '@/lib/feature-flags';
import { PlotDisplay as PlotDisplayOld } from './PlotDisplay.old';
import { PlotDisplayNew } from './PlotDisplay.new';

export function PlotDisplay(props) {
  const useNewVersion = useFeatureFlag('new-plot-display');
  
  if (useNewVersion) {
    return <PlotDisplayNew {...props} />;
  }
  
  return <PlotDisplayOld {...props} />;
}
```

#### 5. Add Migration Tests
```typescript
// components/PlotDisplay.migration.test.tsx
describe('PlotDisplay Migration', () => {
  it('maintains feature parity', () => {
    const props = {
      data: mockData,
      config: mockConfig,
    };
    
    // Render both versions
    const { container: oldContainer } = render(
      <PlotDisplayOld {...props} />
    );
    const { container: newContainer } = render(
      <PlotDisplayNew {...props} />
    );
    
    // Check all interactive elements exist
    const oldButtons = oldContainer.querySelectorAll('button');
    const newButtons = newContainer.querySelectorAll('button');
    expect(newButtons).toHaveLength(oldButtons.length);
    
    // Check content matches
    expect(newContainer).toHaveTextContent(config.title);
    expect(newContainer).toHaveTextContent(config.description);
  });
  
  it('improves accessibility', async () => {
    const { container } = render(<PlotDisplayNew {...mockProps} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
  
  it('maintains or improves performance', () => {
    const oldMetrics = measureComponent(<PlotDisplayOld {...mockProps} />);
    const newMetrics = measureComponent(<PlotDisplayNew {...mockProps} />);
    
    expect(newMetrics.renderTime).toBeLessThanOrEqual(oldMetrics.renderTime * 1.1);
  });
});
```

### CSS Migration Strategy

#### From CSS to Utility Classes
```typescript
// OLD: Using CSS classes
<div className="plot-header">
  <h2 className="plot-title">{title}</h2>
</div>

// CSS
.plot-header {
  display: flex;
  justify-content: space-between;
  padding: 1rem;
  border-bottom: 1px solid #e5e5e5;
}

.plot-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #333;
}

// NEW: Using layout primitives + utilities
<Stack 
  direction="horizontal" 
  justify="space-between" 
  className="p-4 border-b border-gray-200"
>
  <Heading level={2} className="text-gray-800">
    {title}
  </Heading>
</Stack>
```

#### Scoped Styles When Needed
```typescript
// For complex, component-specific styles
import styles from './PlotDisplay.module.css';

<div className={cn(
  'relative overflow-hidden',
  styles.customAnimation
)}>
  {/* Content */}
</div>
```

---

## Testing Procedures

### Test Development Workflow

#### 1. Write Tests First (TDD)
```typescript
// Start with the test
describe('NewFeature', () => {
  it('should handle user input', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    
    render(<NewFeature onSubmit={onSubmit} />);
    
    await user.type(screen.getByLabelText('Name'), 'John Doe');
    await user.click(screen.getByText('Submit'));
    
    expect(onSubmit).toHaveBeenCalledWith({ name: 'John Doe' });
  });
});

// Then implement
export function NewFeature({ onSubmit }) {
  const [name, setName] = useState('');
  
  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      onSubmit({ name });
    }}>
      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <button type="submit">Submit</button>
    </form>
  );
}
```

#### 2. Test Structure
```typescript
describe('ComponentName', () => {
  // Setup
  let mockProps;
  
  beforeEach(() => {
    mockProps = {
      data: createMockData(),
      onEvent: jest.fn(),
    };
  });
  
  // Group by functionality
  describe('Rendering', () => {
    it('renders without crashing', () => {});
    it('displays correct initial state', () => {});
  });
  
  describe('User Interactions', () => {
    it('handles click events', () => {});
    it('validates input correctly', () => {});
  });
  
  describe('Edge Cases', () => {
    it('handles empty data gracefully', () => {});
    it('shows error state appropriately', () => {});
  });
  
  describe('Accessibility', () => {
    it('has no violations', () => {});
    it('supports keyboard navigation', () => {});
  });
});
```

### Running Tests

#### Local Development
```bash
# Run all tests
npm test

# Run tests in watch mode (recommended during development)
npm run test:watch

# Run tests for a specific file
npm test -- PlotDisplay.test

# Run tests matching a pattern
npm test -- --testNamePattern="should handle"

# Run with coverage
npm run test:coverage

# Update snapshots
npm test -- -u
```

#### Pre-commit Testing
```bash
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run tests for changed files
npm run test:staged

# Run linting
npm run lint:staged

# Type checking
npm run type-check
```

#### CI Testing
```yaml
# GitHub Actions example
- name: Run tests
  run: |
    npm run test:ci
    npm run test:integration
    npm run test:visual
```

### Debugging Tests

#### Using VS Code Debugger
```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Debug Jest Tests",
      "runtimeExecutable": "${workspaceFolder}/node_modules/.bin/jest",
      "args": [
        "--runInBand",
        "--no-coverage",
        "${relativeFile}"
      ],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    }
  ]
}
```

#### Debug Utilities
```typescript
// test/utils/debug.ts
export function debugComponent(component: ReactWrapper | HTMLElement) {
  console.log('=== Component Debug ===');
  console.log('HTML:', component.innerHTML || component.html());
  console.log('Props:', component.props?.() || 'N/A');
  console.log('State:', component.state?.() || 'N/A');
  console.log('====================');
}

export function logRenderCycles(componentName: string) {
  let renderCount = 0;
  
  return function LogRender(props: any) {
    renderCount++;
    console.log(`${componentName} rendered ${renderCount} times`);
    console.log('Props:', props);
    return null;
  };
}

// Usage in tests
it('debugging example', () => {
  const { container } = render(
    <>
      <LogRender name="test" />
      <MyComponent />
    </>
  );
  
  debugComponent(container);
});
```

---

## Performance Optimization

### Optimization Techniques

#### 1. Component Memoization
```typescript
// Memoize expensive components
export const ExpensiveComponent = memo(({ data, options }) => {
  const processedData = useMemo(
    () => processData(data, options),
    [data, options]
  );
  
  return <DataVisualization data={processedData} />;
}, (prevProps, nextProps) => {
  // Custom comparison for better performance
  return (
    prevProps.data === nextProps.data &&
    deepEqual(prevProps.options, nextProps.options)
  );
});
```

#### 2. Virtual Scrolling
```typescript
import { useVirtualizer } from '@tanstack/react-virtual';

export function VirtualList({ items }) {
  const parentRef = useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 5,
  });
  
  return (
    <div ref={parentRef} className="h-[400px] overflow-auto">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualItem.size}px`,
              transform: `translateY(${virtualItem.start}px)`,
            }}
          >
            {items[virtualItem.index]}
          </div>
        ))}
      </div>
    </div>
  );
}
```

#### 3. Code Splitting
```typescript
// Route-based splitting
const PlotPage = lazy(() => import('./pages/PlotPage'));

// Component-based splitting
const HeavyComponent = lazy(() => 
  import('./components/HeavyComponent')
    .then(module => ({ default: module.HeavyComponent }))
);

// With loading boundary
<Suspense fallback={<LoadingSpinner />}>
  <HeavyComponent />
</Suspense>
```

#### 4. Image Optimization
```typescript
import Image from 'next/image';

export function OptimizedImage({ src, alt, priority = false }) {
  return (
    <Image
      src={src}
      alt={alt}
      width={800}
      height={600}
      priority={priority}
      loading={priority ? 'eager' : 'lazy'}
      placeholder="blur"
      blurDataURL="data:image/jpeg;base64,..."
      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
    />
  );
}
```

### Performance Monitoring

#### Custom Performance Hook
```typescript
// hooks/usePerformanceMonitor.ts
export function usePerformanceMonitor(componentName: string) {
  useEffect(() => {
    const startTime = performance.now();
    
    // Mark component mount
    performance.mark(`${componentName}-mount-start`);
    
    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      // Mark component unmount
      performance.mark(`${componentName}-mount-end`);
      
      // Measure lifecycle
      performance.measure(
        `${componentName}-lifecycle`,
        `${componentName}-mount-start`,
        `${componentName}-mount-end`
      );
      
      // Log if threshold exceeded
      if (duration > 100) {
        console.warn(
          `${componentName} lifecycle exceeded threshold: ${duration}ms`
        );
      }
      
      // Send to analytics
      if (window.analytics) {
        window.analytics.track('component_performance', {
          component: componentName,
          duration,
          timestamp: new Date().toISOString(),
        });
      }
    };
  }, [componentName]);
}
```

#### Web Vitals Tracking
```typescript
// lib/vitals.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export function reportWebVitals() {
  getCLS(metric => sendToAnalytics('CLS', metric));
  getFID(metric => sendToAnalytics('FID', metric));
  getFCP(metric => sendToAnalytics('FCP', metric));
  getLCP(metric => sendToAnalytics('LCP', metric));
  getTTFB(metric => sendToAnalytics('TTFB', metric));
}

function sendToAnalytics(name: string, metric: any) {
  const body = {
    name,
    value: metric.value,
    delta: metric.delta,
    id: metric.id,
    navigationType: metric.navigationType,
  };
  
  // Use `sendBeacon` for reliability
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/api/analytics', JSON.stringify(body));
  } else {
    fetch('/api/analytics', {
      method: 'POST',
      body: JSON.stringify(body),
      keepalive: true,
    });
  }
}
```

---

## Debugging Tools

### Layout Debugger Component
```typescript
// components/debug/LayoutDebugger.tsx
export function LayoutDebugger({ children }: { children: React.ReactNode }) {
  const [enabled, setEnabled] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [showSpacing, setShowSpacing] = useState(false);
  
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'L') {
        setEnabled(prev => !prev);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);
  
  if (!enabled) return <>{children}</>;
  
  return (
    <div className="relative">
      {/* Debug Controls */}
      <div className="fixed top-4 right-4 z-50 bg-white shadow-lg rounded-lg p-4">
        <h3 className="font-bold mb-2">Layout Debugger</h3>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
          />
          Show Grid
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showSpacing}
            onChange={(e) => setShowSpacing(e.target.checked)}
          />
          Show Spacing
        </label>
      </div>
      
      {/* Debug Overlay */}
      <div
        className={cn(
          'absolute inset-0 pointer-events-none z-40',
          showGrid && 'debug-grid',
          showSpacing && 'debug-spacing'
        )}
      />
      
      {/* Original Content */}
      {children}
    </div>
  );
}
```

### CSS Debug Utilities
```css
/* styles/debug.css */
.debug-grid {
  background-image: 
    repeating-linear-gradient(
      0deg,
      rgba(255, 0, 0, 0.1),
      rgba(255, 0, 0, 0.1) 1px,
      transparent 1px,
      transparent 8px
    ),
    repeating-linear-gradient(
      90deg,
      rgba(255, 0, 0, 0.1),
      rgba(255, 0, 0, 0.1) 1px,
      transparent 1px,
      transparent 8px
    );
}

.debug-spacing * {
  outline: 1px solid rgba(0, 255, 0, 0.3) !important;
  outline-offset: -1px;
}

.debug-spacing *::before {
  content: attr(data-spacing);
  position: absolute;
  top: 0;
  right: 0;
  font-size: 10px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 2px 4px;
  pointer-events: none;
}
```

### React DevTools Profiler
```typescript
// Enable profiling in development
if (process.env.NODE_ENV === 'development') {
  if (typeof window !== 'undefined') {
    window.React = React;
    
    // Log render reasons
    if (process.env.NEXT_PUBLIC_LOG_RENDERS === 'true') {
      const whyDidYouRender = require('@welldone-software/why-did-you-render');
      whyDidYouRender(React, {
        trackAllPureComponents: true,
        logOnDifferentValues: true,
      });
    }
  }
}

// Mark components for tracking
PlotDisplay.whyDidYouRender = {
  logOnDifferentValues: true,
  customName: 'PlotDisplay'
};
```

### Performance Timeline Visualizer
```typescript
// components/debug/PerformanceTimeline.tsx
export function PerformanceTimeline() {
  const [entries, setEntries] = useState<PerformanceEntry[]>([]);
  
  useEffect(() => {
    const observer = new PerformanceObserver((list) => {
      setEntries(prev => [...prev, ...list.getEntries()]);
    });
    
    observer.observe({ 
      entryTypes: ['measure', 'navigation', 'resource'] 
    });
    
    return () => observer.disconnect();
  }, []);
  
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white shadow-lg p-4 max-h-64 overflow-auto">
      <h4 className="font-bold mb-2">Performance Timeline</h4>
      <div className="space-y-1">
        {entries.map((entry, index) => (
          <div key={index} className="flex items-center gap-2 text-xs">
            <span className="font-mono">{entry.startTime.toFixed(2)}ms</span>
            <span className="font-semibold">{entry.name}</span>
            <span className="text-gray-500">{entry.duration.toFixed(2)}ms</span>
            <div 
              className="bg-blue-500 h-2"
              style={{ 
                width: `${Math.min(entry.duration / 10, 100)}px` 
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Console Helpers
```typescript
// lib/debug-helpers.ts
if (process.env.NODE_ENV === 'development') {
  // Layout debugging
  window.debugLayout = {
    showBounds: () => {
      document.body.classList.add('debug-bounds');
    },
    hideBounds: () => {
      document.body.classList.remove('debug-bounds');
    },
    inspectSpacing: (selector: string) => {
      const element = document.querySelector(selector);
      if (element) {
        const styles = getComputedStyle(element);
        console.table({
          margin: styles.margin,
          padding: styles.padding,
          gap: styles.gap,
          width: styles.width,
          height: styles.height,
        });
      }
    },
    measureReflow: async (callback: () => void) => {
      const start = performance.now();
      const startLayouts = performance.getEntriesByType('layout').length;
      
      await callback();
      
      const end = performance.now();
      const endLayouts = performance.getEntriesByType('layout').length;
      
      console.log({
        duration: `${(end - start).toFixed(2)}ms`,
        layouts: endLayouts - startLayouts,
      });
    },
  };
  
  // Component debugging
  window.debugComponent = {
    logProps: (componentName: string) => {
      const original = console.log;
      console.log = (...args) => {
        if (args[0]?.includes?.(componentName)) {
          original.apply(console, args);
        }
      };
    },
    traceUpdates: () => {
      const ReactDOM = require('react-dom');
      const roots = new Set();
      
      const originalRender = ReactDOM.render;
      ReactDOM.render = (element: any, container: any, ...args: any[]) => {
        roots.add(container);
        return originalRender(element, container, ...args);
      };
      
      setInterval(() => {
        roots.forEach(root => {
          root.style.outline = '3px solid red';
          setTimeout(() => {
            root.style.outline = '';
          }, 500);
        });
      }, 1000);
    },
  };
}
```