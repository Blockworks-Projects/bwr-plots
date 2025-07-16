# Layout System V2

A modern, flexible layout system for the BWR Plots application that replaces the rigid fixed-width layout with a composable, responsive architecture.

## Overview

The new layout system provides:
- **Flexible panels** that can be resized, collapsed, and rearranged
- **Scroll position persistence** across component updates
- **Responsive design** that adapts to different screen sizes
- **Backward compatibility** through adapter components
- **Performance optimizations** with efficient re-rendering

## Quick Start

### Basic Panel Usage

```tsx
import { Panel, ScrollArea } from '@/components/layout/v2';

function MyPanel() {
  return (
    <Panel
      id="my-panel"
      title="My Panel"
      icon="📊"
    >
      <ScrollArea persistKey="my-panel-scroll">
        <div className="space-y-4">
          {/* Your content */}
        </div>
      </ScrollArea>
    </Panel>
  );
}
```

### Flexible Layout

```tsx
import { LayoutContainer, FlexLayout, FlexItem } from '@/components/layout/v2';

function MyLayout() {
  return (
    <LayoutContainer>
      <FlexLayout direction="row" fullHeight>
        <FlexItem basis={320} shrink={false}>
          <Panel title="Sidebar">Sidebar content</Panel>
        </FlexItem>
        <FlexItem flex={1}>
          <Panel title="Main">Main content</Panel>
        </FlexItem>
      </FlexLayout>
    </LayoutContainer>
  );
}
```

### Migration Adapter

```tsx
import { DashboardLayoutAdapter } from '@/components/layout/v2';

function MyPage() {
  return (
    <DashboardLayoutAdapter
      dataPanel={<DataPanel />}
      plotCanvas={<PlotCanvas />}
      configPanel={<ConfigurationPanel />}
      useNewPanels={true} // Enable new layout gradually
    />
  );
}
```

## Components

### LayoutContainer

Root container that manages resize observation and provides layout context.

**Props:**
- `children: ReactNode` - Layout content
- `className?: string` - Additional CSS classes
- `minHeight?: number` - Minimum container height
- `maxHeight?: number` - Maximum container height
- `onResize?: (dimensions) => void` - Resize callback

**Features:**
- ResizeObserver integration
- Layout dimension tracking
- Debug overlay in development

### Panel

Base panel component with header, content, and footer areas.

**Props:**
- `id?: string` - Unique panel identifier
- `title?: string` - Panel title
- `icon?: string | ReactNode` - Panel icon
- `actions?: ReactNode` - Header action buttons
- `footer?: ReactNode` - Footer content
- `scrollable?: boolean` - Enable scrolling (default: true)
- `padding?: boolean | 'sm' | 'md' | 'lg'` - Content padding
- `variant?: 'default' | 'elevated' | 'bordered'` - Visual style
- `legacy?: boolean` - Backward compatibility mode

**Subcomponents:**
- `Panel.Header` - Standalone header component
- `Panel.Content` - Standalone content component
- `Panel.Footer` - Standalone footer component

### FlexLayout

Flexible container for arranging child components.

**Props:**
- `direction?: 'row' | 'column' | 'row-reverse' | 'column-reverse'`
- `wrap?: 'wrap' | 'nowrap' | 'wrap-reverse'`
- `align?: 'start' | 'end' | 'center' | 'stretch' | 'baseline'`
- `justify?: 'start' | 'end' | 'center' | 'between' | 'around' | 'evenly'`
- `gap?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl' | number`
- `fullHeight?: boolean` - Take full height
- `fullWidth?: boolean` - Take full width
- `responsive?: object` - Responsive breakpoint overrides

### FlexItem

Flexible item within a FlexLayout.

**Props:**
- `flex?: boolean | number | string` - Flex value
- `grow?: boolean | number` - Flex grow
- `shrink?: boolean | number` - Flex shrink
- `basis?: string | number` - Flex basis
- `alignSelf?: 'auto' | FlexAlign` - Self alignment
- `order?: number` - Display order

### ScrollArea

Managed scrolling component with position persistence.

**Props:**
- `orientation?: 'vertical' | 'horizontal' | 'both'`
- `persistKey?: string` - Key for scroll position storage
- `onScroll?: (position) => void` - Scroll event handler
- `restorePosition?: boolean` - Auto-restore scroll position
- `showScrollbar?: 'always' | 'auto' | 'hover'`
- `smoothScroll?: boolean` - Enable smooth scrolling

**Features:**
- Automatic scroll position saving/restoring
- Cross-session persistence via sessionStorage
- Scroll event throttling

## Migration Guide

### Phase 1: Enable Feature Flag

Set the feature flag to gradually roll out the new system:

```tsx
// Enable for specific users/pages
localStorage.setItem('bwr-plots-new-layout', 'true');

// Or use environment variable
NEXT_PUBLIC_NEW_LAYOUT=true
```

### Phase 2: Wrap Existing Components

Use adapter components to gradually migrate:

```tsx
import { LegacyPanelAdapter } from '@/components/layout/v2';

// Before
<div className="w-80 h-full overflow-y-auto p-4">
  <DataPanel />
</div>

// After (with adapter)
<LegacyPanelAdapter
  title="Data"
  className="w-80 h-full"
>
  <DataPanel />
</LegacyPanelAdapter>
```

### Phase 3: Full Migration

Replace with new layout components:

```tsx
// Before
<div className="h-full w-full flex">
  <div className="w-80">{dataPanel}</div>
  <div className="flex-1">{plotCanvas}</div>
  <div className="w-96">{configPanel}</div>
</div>

// After
<FlexLayout direction="row" fullHeight>
  <FlexItem basis={320} shrink={false}>
    <Panel title="Data">{dataPanel}</Panel>
  </FlexItem>
  <FlexItem flex={1}>
    <Panel title="Plot">{plotCanvas}</Panel>
  </FlexItem>
  <FlexItem basis={384} shrink={false}>
    <Panel title="Config">{configPanel}</Panel>
  </FlexItem>
</FlexLayout>
```

## Best Practices

### 1. Use Semantic Panel IDs

```tsx
<Panel id="data-panel" title="Data">
  {/* Enables better debugging and persistence */}
</Panel>
```

### 2. Persist Scroll Positions

```tsx
<ScrollArea persistKey="unique-panel-scroll">
  {/* Scroll position survives component updates */}
</ScrollArea>
```

### 3. Define Min/Max Sizes

```tsx
<FlexItem 
  basis={320} 
  shrink={false}
  style={{ minWidth: 240, maxWidth: 480 }}
>
  {/* Prevents panels from becoming too small/large */}
</FlexItem>
```

### 4. Use Progressive Enhancement

```tsx
<MigrationWrapper
  oldComponent={<OldLayout />}
  newComponent={<NewLayout />}
/>
```

### 5. Handle Responsive Design

```tsx
<FlexLayout
  direction="row"
  responsive={{
    sm: { direction: 'column' },
    md: { direction: 'row' }
  }}
>
  {/* Adapts to screen size */}
</FlexLayout>
```

## Performance Considerations

### ResizeObserver Efficiency

The layout system uses ResizeObserver for efficient resize detection:

```tsx
// Debounced resize events prevent performance issues
const [isResizing, setIsResizing] = useState(false);
```

### Scroll Position Caching

Scroll positions are cached in memory and persisted to sessionStorage:

```tsx
// Memory cache for fast access
const scrollPositionCache = new Map();

// SessionStorage for persistence
sessionStorage.setItem('scroll-positions', JSON.stringify(positions));
```

### Minimal Re-renders

Components only re-render when their specific props change:

```tsx
// Uses React.memo and careful dependency arrays
const Panel = React.memo(PanelComponent);
```

## Troubleshooting

### Common Issues

**Scroll position not persisting:**
- Ensure unique `persistKey` prop
- Check that `restorePosition` is true (default)

**Layout not responsive:**
- Verify `LayoutContainer` wraps the layout
- Check responsive prop configuration

**Panels not resizing:**
- Ensure proper flex properties on FlexItem
- Check min/max constraints

**Performance issues:**
- Reduce unnecessary re-renders with React.memo
- Use stable keys for dynamic content

### Debug Mode

Development builds include debug overlays:

```tsx
// Shows layout dimensions
<LayoutContainer>
  {/* Debug info appears in top-left corner */}
</LayoutContainer>
```

### Layout Debugger

Use the existing layout debugger for troubleshooting:

```tsx
import { LayoutDebugger } from '@/components/debug/LayoutDebugger';

// Add to your page
<LayoutDebugger />
```

## Testing

### Unit Tests

```tsx
import { render, screen } from '@testing-library/react';
import { Panel } from '@/components/layout/v2';

test('renders panel with title', () => {
  render(<Panel title="Test">Content</Panel>);
  expect(screen.getByText('Test')).toBeInTheDocument();
});
```

### Integration Tests

```tsx
test('maintains scroll position', async () => {
  const { rerender } = render(
    <ScrollArea persistKey="test">
      <div style={{ height: '1000px' }}>Content</div>
    </ScrollArea>
  );
  
  // Scroll, update component, verify position maintained
});
```

## Examples

See `examples.tsx` for complete working examples:

- Basic panel migration
- Flexible dashboard layout
- Progressive enhancement
- Responsive design
- Adapter usage

## Migration Checklist

- [ ] Enable feature flag
- [ ] Wrap existing components with adapters
- [ ] Test layout in different screen sizes
- [ ] Verify scroll position persistence
- [ ] Update component tests
- [ ] Performance test with real data
- [ ] Remove old layout code
- [ ] Update documentation

## Support

For questions or issues:
1. Check this documentation
2. Review examples in `examples.tsx`
3. Use the layout debugger for visual inspection
4. Run tests to verify behavior
5. Check console for debug information