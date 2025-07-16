# BWR Tools Layout System Migration Guide

## Overview

This guide helps you migrate from the existing `DashboardLayout` to the new comprehensive layout system while maintaining backward compatibility.

## Migration Strategy

### Phase 1: Immediate Usage (Zero Breaking Changes)

The new layout system is fully compatible with existing code. You can start using new components alongside the current `DashboardLayout`.

```tsx
// Existing code continues to work
import { DashboardLayout } from '@/components/layout';

// New components are available
import { ViewportProvider, AppShell, FlexLayout, Panel } from '@/components/layout';
```

### Phase 2: Gradual Migration

#### Current DashboardLayout Pattern:
```tsx
<DashboardLayout
  dataPanel={<DataPanel />}
  plotCanvas={<PlotCanvas />}
  configPanel={<ConfigPanel />}
/>
```

#### New FlexLayout Pattern:
```tsx
<ViewportProvider>
  <AppShell>
    <FlexLayout orientation="horizontal" resizable>
      <FlexItem size="320px" minSize="280px" maxSize="400px">
        <Panel title="Data" bordered>
          <DataPanel />
        </Panel>
      </FlexItem>
      
      <FlexItem grow={1}>
        <Panel title="Visualization" bordered>
          <PlotCanvas />
        </Panel>
      </FlexItem>
      
      <FlexItem size="384px" minSize="320px" maxSize="480px">
        <Panel title="Configuration" bordered>
          <ConfigPanel />
        </Panel>
      </FlexItem>
    </FlexLayout>
  </AppShell>
</ViewportProvider>
```

## Key Benefits of Migration

### 1. Enhanced Responsiveness
```tsx
// Old: Fixed layout
<div className="w-80 flex-shrink-0">

// New: Responsive with breakpoints
<FlexItem size={{ sm: "100%", lg: "320px" }} minSize="280px">
```

### 2. Better Performance
```tsx
// Old: Basic scrolling
<div className="overflow-y-auto">

// New: Optimized scroll with virtualization
<ScrollArea virtualizer={{ itemCount: 1000, itemSize: 40 }}>
```

### 3. Consistent Viewport Management
```tsx
// Old: Manual window size tracking
const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });

// New: Built-in viewport context
const { width, height, breakpoint, isMobile } = useViewport();
```

## Component Migration Map

| Old Component | New Component | Migration Effort |
|---------------|---------------|------------------|
| `DashboardLayout` | `AppShell + FlexLayout` | Medium |
| Manual flex containers | `FlexLayout + FlexItem` | Low |
| CSS Grid layouts | `Grid + GridItem` | Low |
| Scroll containers | `ScrollArea` | Low |
| Panel headers | `Panel` | Low |

## Step-by-Step Migration

### Step 1: Add ViewportProvider
Wrap your application root with the ViewportProvider:

```tsx
// app/layout.tsx
import { ViewportProvider } from '@/components/layout';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ViewportProvider>
          {children}
        </ViewportProvider>
      </body>
    </html>
  );
}
```

### Step 2: Replace Basic Layouts
Start with simple layouts first:

```tsx
// Before
<div className="flex h-full">
  <div className="w-64 bg-gray-100">Sidebar</div>
  <div className="flex-1">Main</div>
</div>

// After
<FlexLayout orientation="horizontal">
  <FlexItem size="256px">
    <Panel>Sidebar</Panel>
  </FlexItem>
  <FlexItem grow={1}>
    <Panel>Main</Panel>
  </FlexItem>
</FlexLayout>
```

### Step 3: Enhance with Advanced Features
Add responsive behavior and performance optimizations:

```tsx
// Add responsive sizing
<FlexItem size={{ sm: "100%", md: "256px" }}>

// Add scroll optimization
<ScrollArea height="100%" scrollbarWidth="thin">

// Add resize handles
<FlexLayout orientation="horizontal" resizable>
```

### Step 4: Migrate Complex Layouts
Replace the main DashboardLayout:

```tsx
// Create a new layout component
export function NewDashboardLayout({ children }: { children: React.ReactNode }) {
  const { isMobile } = useViewport();
  
  return (
    <AppShell
      header={<Header />}
      sidebar={isMobile ? undefined : <Sidebar />}
    >
      <FlexLayout 
        orientation={isMobile ? "vertical" : "horizontal"} 
        resizable={!isMobile}
      >
        {children}
      </FlexLayout>
    </AppShell>
  );
}
```

## CSS Migration

### Import New Styles
Add the layout system CSS to your globals.css:

```css
/* globals.css */
@import "tailwindcss";
@import "./components/layout/styles/variables.css";

/* Your existing styles remain unchanged */
```

### CSS Variable Updates
The new system uses CSS custom properties for theming:

```css
/* Use new layout variables */
.my-component {
  padding: var(--layout-spacing-md);
  border-radius: var(--layout-radius-md);
  box-shadow: var(--layout-shadow-md);
}
```

## Performance Considerations

### Scroll Optimization
```tsx
// For large lists, enable virtualization
<ScrollArea 
  virtualizer={{
    itemCount: data.length,
    itemSize: 48,
    overscan: 5
  }}
>
  {virtualItems.map(item => (
    <VirtualItem key={item.key} />
  ))}
</ScrollArea>
```

### Resize Performance
```tsx
// Debounce resize events
const { width } = useViewport(); // Already optimized internally

// Or use the resize observer hook
useResizeObserver({
  ref: elementRef,
  onResize: debounce(handleResize, 100)
});
```

## Troubleshooting

### Common Issues

1. **Layout jumping on resize**
   ```tsx
   // Solution: Use stable sizes
   <FlexItem minSize="280px" maxSize="400px">
   ```

2. **Scroll not working**
   ```tsx
   // Solution: Ensure container has explicit height
   <ScrollArea height="400px"> // or height="100%"
   ```

3. **Responsive not updating**
   ```tsx
   // Solution: Ensure ViewportProvider is at root level
   <ViewportProvider debounceMs={50}>
   ```

### Debug Tools

Enable layout debugging in development:

```tsx
import { LayoutDebugger } from '@/components/debug';

// Add to your layout during development
{process.env.NODE_ENV === 'development' && <LayoutDebugger />}
```

## Backward Compatibility

The migration is designed to be non-breaking:

- Existing `DashboardLayout` continues to work
- CSS classes remain unchanged
- No TypeScript breaking changes
- Gradual adoption path

## Next Steps

1. **Start Small**: Begin with new components in isolated areas
2. **Test Thoroughly**: Use the examples to validate behavior
3. **Migrate Gradually**: Replace layouts one component at a time
4. **Monitor Performance**: Use browser dev tools to verify improvements
5. **Gather Feedback**: Collect team feedback during migration

## Resources

- [Layout System Documentation](./README.md)
- [Component Examples](./examples.tsx)
- [TypeScript Definitions](./types/index.ts)
- [Performance Guide](./utils/performance.ts)