/**
 * Layout Migration Tests
 * ---
 * bwr-plots/frontend/src/components/layout/v2/__tests__/layout-migration.test.tsx
 * ---
 * Comprehensive tests for layout system migration
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

import {
  LayoutContainer,
  Panel,
  FlexLayout,
  FlexItem,
  ScrollArea,
  LegacyPanelAdapter,
  DashboardLayoutAdapter,
  MigrationWrapper,
  useNewLayoutSystem
} from '../index';

// Mock localStorage for feature flag tests
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Layout Container Tests                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('LayoutContainer', () => {
  it('should render children correctly', () => {
    render(
      <LayoutContainer>
        <div data-testid="child">Test Content</div>
      </LayoutContainer>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it('should set up ResizeObserver', () => {
    render(<LayoutContainer><div>Content</div></LayoutContainer>);
    expect(global.ResizeObserver).toHaveBeenCalled();
  });

  it('should apply custom className', () => {
    const { container } = render(
      <LayoutContainer className="custom-class">
        <div>Content</div>
      </LayoutContainer>
    );

    expect(container.firstChild).toHaveClass('layout-container', 'custom-class');
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Component Tests                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('Panel', () => {
  it('should render basic panel without header', () => {
    render(
      <Panel>
        <div data-testid="content">Panel Content</div>
      </Panel>
    );

    expect(screen.getByTestId('content')).toBeInTheDocument();
    expect(screen.queryByRole('banner')).not.toBeInTheDocument();
  });

  it('should render panel with title and icon', () => {
    render(
      <Panel title="Test Panel" icon="🔧">
        <div>Content</div>
      </Panel>
    );

    expect(screen.getByText('Test Panel')).toBeInTheDocument();
    expect(screen.getByText('🔧')).toBeInTheDocument();
  });

  it('should render panel with actions', () => {
    const actions = <button data-testid="action-btn">Action</button>;
    
    render(
      <Panel title="Test Panel" actions={actions}>
        <div>Content</div>
      </Panel>
    );

    expect(screen.getByTestId('action-btn')).toBeInTheDocument();
  });

  it('should render panel with footer', () => {
    const footer = <div data-testid="footer">Footer Content</div>;
    
    render(
      <Panel footer={footer}>
        <div>Content</div>
      </Panel>
    );

    expect(screen.getByTestId('footer')).toBeInTheDocument();
  });

  it('should apply variant classes', () => {
    const { container } = render(
      <Panel variant="elevated">
        <div>Content</div>
      </Panel>
    );

    expect(container.firstChild).toHaveClass('bg-[var(--color-bg-elevated)]');
  });

  it('should work in legacy mode', () => {
    const { container } = render(
      <Panel legacy className="legacy-class">
        <div>Legacy Content</div>
      </Panel>
    );

    expect(container.firstChild).toHaveClass('legacy-class');
    expect(screen.getByText('Legacy Content')).toBeInTheDocument();
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ FlexLayout Tests                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('FlexLayout', () => {
  it('should render with default flex row layout', () => {
    const { container } = render(
      <FlexLayout>
        <div>Item 1</div>
        <div>Item 2</div>
      </FlexLayout>
    );

    expect(container.firstChild).toHaveClass('flex', 'flex-row');
  });

  it('should apply direction correctly', () => {
    const { container } = render(
      <FlexLayout direction="column">
        <div>Item</div>
      </FlexLayout>
    );

    expect(container.firstChild).toHaveClass('flex-col');
  });

  it('should apply gap classes', () => {
    const { container } = render(
      <FlexLayout gap="lg">
        <div>Item</div>
      </FlexLayout>
    );

    expect(container.firstChild).toHaveClass('gap-6');
  });

  it('should apply custom gap as style', () => {
    const { container } = render(
      <FlexLayout gap={16}>
        <div>Item</div>
      </FlexLayout>
    );

    expect(container.firstChild).toHaveStyle({ gap: '16px' });
  });

  it('should handle fullHeight and fullWidth', () => {
    const { container } = render(
      <FlexLayout fullHeight fullWidth>
        <div>Item</div>
      </FlexLayout>
    );

    expect(container.firstChild).toHaveClass('h-full', 'w-full');
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ FlexItem Tests                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('FlexItem', () => {
  it('should render with basic props', () => {
    render(
      <FlexItem>
        <div data-testid="content">Item Content</div>
      </FlexItem>
    );

    expect(screen.getByTestId('content')).toBeInTheDocument();
  });

  it('should apply flex=true as flex-1 class', () => {
    const { container } = render(
      <FlexItem flex>
        <div>Content</div>
      </FlexItem>
    );

    expect(container.firstChild).toHaveClass('flex-1');
  });

  it('should apply flex=false as flex-none class', () => {
    const { container } = render(
      <FlexItem flex={false}>
        <div>Content</div>
      </FlexItem>
    );

    expect(container.firstChild).toHaveClass('flex-none');
  });

  it('should apply custom flex value as style', () => {
    const { container } = render(
      <FlexItem flex="2 1 auto">
        <div>Content</div>
      </FlexItem>
    );

    expect(container.firstChild).toHaveStyle({ flex: '2 1 auto' });
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ ScrollArea Tests                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('ScrollArea', () => {
  beforeEach(() => {
    // Mock sessionStorage
    Object.defineProperty(window, 'sessionStorage', {
      value: {
        getItem: jest.fn(),
        setItem: jest.fn(),
      },
      writable: true,
    });
  });

  it('should render children in scrollable container', () => {
    render(
      <ScrollArea>
        <div data-testid="scrollable-content">Scrollable Content</div>
      </ScrollArea>
    );

    expect(screen.getByTestId('scrollable-content')).toBeInTheDocument();
  });

  it('should apply orientation classes', () => {
    const { container } = render(
      <ScrollArea orientation="horizontal">
        <div>Content</div>
      </ScrollArea>
    );

    expect(container.firstChild).toHaveClass('overflow-x-auto', 'overflow-y-hidden');
  });

  it('should handle scroll events', () => {
    const onScroll = jest.fn();
    
    render(
      <ScrollArea onScroll={onScroll}>
        <div style={{ height: '200px' }}>Tall Content</div>
      </ScrollArea>
    );

    const scrollArea = screen.getByRole('generic');
    fireEvent.scroll(scrollArea, { target: { scrollTop: 100 } });

    expect(onScroll).toHaveBeenCalled();
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Adapter Tests                                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('LegacyPanelAdapter', () => {
  it('should wrap children in Panel component', () => {
    render(
      <LegacyPanelAdapter title="Legacy Panel">
        <div data-testid="legacy-content">Legacy Content</div>
      </LegacyPanelAdapter>
    );

    expect(screen.getByText('Legacy Panel')).toBeInTheDocument();
    expect(screen.getByTestId('legacy-content')).toBeInTheDocument();
  });

  it('should separate wrapper and panel classes', () => {
    const { container } = render(
      <LegacyPanelAdapter 
        className="w-80 flex-shrink-0 h-full"
        wrapperClassName="border-test"
      >
        <div>Content</div>
      </LegacyPanelAdapter>
    );

    // Wrapper should have width/flex classes
    expect(container.firstChild).toHaveClass('w-80', 'flex-shrink-0', 'h-full', 'border-test');
  });
});

describe('DashboardLayoutAdapter', () => {
  const mockPanels = {
    dataPanel: <div data-testid="data">Data Panel</div>,
    plotCanvas: <div data-testid="plot">Plot Canvas</div>,
    configPanel: <div data-testid="config">Config Panel</div>
  };

  it('should render old layout by default', () => {
    render(<DashboardLayoutAdapter {...mockPanels} />);

    expect(screen.getByTestId('data')).toBeInTheDocument();
    expect(screen.getByTestId('plot')).toBeInTheDocument();
    expect(screen.getByTestId('config')).toBeInTheDocument();
  });

  it('should use new layout when enabled', () => {
    render(<DashboardLayoutAdapter {...mockPanels} useNewPanels />);

    // Should still render all panels
    expect(screen.getByTestId('data')).toBeInTheDocument();
    expect(screen.getByTestId('plot')).toBeInTheDocument();
    expect(screen.getByTestId('config')).toBeInTheDocument();
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Migration Wrapper Tests                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('MigrationWrapper', () => {
  it('should render old component by default', () => {
    mockLocalStorage.getItem.mockReturnValue('false');
    
    render(
      <MigrationWrapper
        oldComponent={<div data-testid="old">Old Component</div>}
        newComponent={<div data-testid="new">New Component</div>}
      />
    );

    expect(screen.getByTestId('old')).toBeInTheDocument();
    expect(screen.queryByTestId('new')).not.toBeInTheDocument();
  });

  it('should render new component when forced', () => {
    render(
      <MigrationWrapper
        oldComponent={<div data-testid="old">Old Component</div>}
        newComponent={<div data-testid="new">New Component</div>}
        forceNew
      />
    );

    expect(screen.getByTestId('new')).toBeInTheDocument();
    expect(screen.queryByTestId('old')).not.toBeInTheDocument();
  });
});

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Integration Tests                                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

describe('Layout Integration', () => {
  it('should handle nested layout components', () => {
    render(
      <LayoutContainer>
        <FlexLayout direction="column">
          <FlexItem>
            <Panel title="Header Panel">
              <div data-testid="header">Header Content</div>
            </Panel>
          </FlexItem>
          <FlexItem flex>
            <FlexLayout direction="row">
              <FlexItem basis={300}>
                <Panel title="Sidebar">
                  <ScrollArea>
                    <div data-testid="sidebar">Sidebar Content</div>
                  </ScrollArea>
                </Panel>
              </FlexItem>
              <FlexItem flex>
                <Panel title="Main">
                  <div data-testid="main">Main Content</div>
                </Panel>
              </FlexItem>
            </FlexLayout>
          </FlexItem>
        </FlexLayout>
      </LayoutContainer>
    );

    expect(screen.getByTestId('header')).toBeInTheDocument();
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
    expect(screen.getByTestId('main')).toBeInTheDocument();
    expect(screen.getByText('Header Panel')).toBeInTheDocument();
    expect(screen.getByText('Sidebar')).toBeInTheDocument();
    expect(screen.getByText('Main')).toBeInTheDocument();
  });

  it('should maintain scroll position during updates', async () => {
    const TestScrollComponent = () => {
      const [items, setItems] = React.useState([1, 2, 3]);

      return (
        <ScrollArea persistKey="test-scroll">
          <div>
            {items.map(item => (
              <div key={item} style={{ height: '100px' }}>
                Item {item}
              </div>
            ))}
          </div>
          <button
            data-testid="add-item"
            onClick={() => setItems(prev => [...prev, prev.length + 1])}
          >
            Add Item
          </button>
        </ScrollArea>
      );
    };

    render(<TestScrollComponent />);

    const addButton = screen.getByTestId('add-item');
    fireEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText('Item 4')).toBeInTheDocument();
    });
  });
});