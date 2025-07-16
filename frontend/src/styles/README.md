# BWR Plots Design System

A comprehensive CSS architecture and design token system following the 60-30-10 design principle with a purple (#5B35D5) theme.

## 📁 Architecture Overview

```
src/styles/
├── tokens/
│   └── index.css           # Core design tokens
├── themes/
│   ├── dark.css           # Dark theme (default)
│   └── light.css          # Light theme
├── layout/
│   └── primitives.module.css  # Layout primitive classes
├── components/
│   └── scrollbar.css      # Custom scrollbar styling
├── utilities/
│   └── layout.ts          # TypeScript utility functions
└── README.md              # This file
```

## 🎨 Design Tokens

### Color System (60-30-10 Principle)

#### 10% - Primary Colors (Accents & CTAs)
```css
--color-primary: #5B35D5        /* Main purple */
--color-primary-hover: #6D43DD   /* Hover state */
--color-primary-active: #4E2CC0  /* Active state */
```

#### 60% - Surface Colors (Main backgrounds)
```css
--color-background: var(--gray-950)  /* Main background */
--color-surface-1: var(--gray-900)   /* Cards, containers */
--color-surface-2: var(--gray-800)   /* Elevated surfaces */
--color-surface-3: var(--dark-300)   /* Highest elevation */
```

#### 30% - Border Colors (Supporting elements)
```css
--color-border-default: var(--gray-700)  /* Default borders */
--color-border-light: var(--gray-600)    /* Subtle borders */
--color-border-strong: var(--gray-500)   /* Emphasis borders */
```

### Spacing Scale (4px base)
```css
--space-1: 0.25rem    /* 4px */
--space-2: 0.5rem     /* 8px */
--space-4: 1rem       /* 16px */
--space-6: 1.5rem     /* 24px */
--space-8: 2rem       /* 32px */
--space-12: 3rem      /* 48px */
```

### Typography Scale (Perfect Fourth - 1.333)
```css
--text-xs: 0.75rem    /* 12px */
--text-sm: 0.875rem   /* 14px */
--text-base: 1rem     /* 16px */
--text-lg: 1.125rem   /* 18px */
--text-xl: 1.25rem    /* 20px */
--text-2xl: 1.5rem    /* 24px */
```

## 🧩 Layout Primitives

### CSS Modules Approach
We use CSS Modules for component-specific styling with BEM-like naming:

```css
/* Box Primitive */
.box_padding_md { padding: var(--space-4); }
.box_margin_auto { margin: auto; }

/* Flex Primitive */
.flex_justifyCenter { justify-content: center; }
.flex_alignCenter { align-items: center; }
.flex_gap_md { gap: var(--space-4); }

/* Grid Primitive */
.grid_cols_3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid_gap_lg { gap: var(--space-6); }
```

### TypeScript Utilities
```typescript
import { flex, grid, spacing } from '@/styles/utilities/layout';

// Dynamic flex styles
const flexStyles = flex({
  direction: 'column',
  justify: 'center',
  align: 'center',
  gap: 'md'
});

// Dynamic grid styles
const gridStyles = grid({
  columns: 3,
  gap: 'lg',
  alignItems: 'center'
});
```

## 🎯 Component Patterns

### Buttons
```css
.btn                  /* Base button styles */
.btn-primary          /* Purple primary button */
.btn-secondary        /* Neutral secondary button */
.btn-ghost            /* Transparent button */
.btn-sm | .btn-lg     /* Size variants */
```

### Cards
```css
.card                 /* Base card */
.card-elevated        /* Higher elevation */
.card-interactive     /* Clickable card */
.card-header          /* Card header section */
.card-footer          /* Card footer section */
```

### Forms
```css
input, select, textarea  /* Auto-styled form elements */
:focus                   /* Purple focus ring */
:disabled                /* Disabled state styling */
```

## 🌙 Theming System

### Theme Switching
```typescript
import { useTheme } from '@/contexts/ThemeContext';

function MyComponent() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  
  return (
    <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
      Current theme: {resolvedTheme}
    </button>
  );
}
```

### Theme Attributes
```css
[data-theme="dark"] { /* Dark theme variables */ }
[data-theme="light"] { /* Light theme variables */ }
```

### SSR Theme Support
Add to your document head to prevent flash:
```html
<script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
```

## 📱 Responsive Design

### Breakpoints
```css
--screen-xs: 475px
--screen-sm: 640px
--screen-md: 768px
--screen-lg: 1024px
--screen-xl: 1280px
--screen-2xl: 1536px
--screen-3xl: 1920px
```

### Responsive Utilities
```css
.hide_md           /* Hide on medium screens and up */
.show_lg           /* Show on large screens and up */
.grid_md_cols_3    /* 3 columns on medium screens and up */
.flex_lg_row       /* Row direction on large screens and up */
```

### TypeScript Responsive Values
```typescript
import { getResponsiveValue, type ResponsiveValue } from '@/styles/utilities/layout';

type Props = {
  gap: ResponsiveValue<SpacingScale>;
};

// Usage: gap={{ xs: 'sm', lg: 'lg' }}
```

## 🎭 Animation & Transitions

### Duration Tokens
```css
--duration-150: 150ms
--duration-200: 200ms
--duration-300: 300ms
--duration-500: 500ms
```

### Easing Functions
```css
--ease-linear: linear
--ease-in: cubic-bezier(0.4, 0, 1, 1)
--ease-out: cubic-bezier(0, 0, 0.2, 1)
--ease-in-out: cubic-bezier(0.4, 0, 0.2, 1)
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1)
```

## 📏 Scrollbars

### Custom Scrollbar Variants
```css
.scrollbar-thin        /* Thin scrollbar */
.scrollbar-none        /* Hidden scrollbar */
.scrollbar-overlay     /* macOS-style overlay */
.scrollbar-purple      /* Purple themed scrollbar */
```

### Smooth Scrolling
```css
.scroll-smooth         /* Enable smooth scrolling */
.scroll-snap-y         /* Vertical scroll snapping */
.scroll-gpu            /* Hardware acceleration */
```

## 🚀 Performance Optimizations

### Build-time Optimizations
- **PostCSS**: Custom property optimization
- **Tailwind CSS v4**: JIT compilation
- **CSS Modules**: Scoped styles with build-time hashing

### Runtime Optimizations
- **CSS Custom Properties**: Theme switching without runtime CSS generation
- **Hardware Acceleration**: GPU-optimized scrolling and transitions
- **Contain**: Layout and style containment for better performance

### Bundle Size
- **Tree Shaking**: Unused CSS automatically removed
- **Critical CSS**: Above-the-fold styles inlined
- **CSS Splitting**: Component-level CSS chunks

## 🎨 Purple Theme Implementation

### 60-30-10 Application
- **60%**: Dark neutral backgrounds (`#09090B`, `#18181B`, `#27272A`)
- **30%**: Subtle borders and dividers (`#3F3F46`, `#52525B`)
- **10%**: Purple accents and CTAs (`#5B35D5`, `#6D43DD`)

### Purple Variations
```css
/* Light purple tints for backgrounds */
--color-primary-subtle: rgba(91, 53, 213, 0.1)
--color-primary-border: rgba(91, 53, 213, 0.3)

/* Purple shadows for elevation */
--shadow-purple-sm: 0 1px 3px 0 rgb(91 33 182 / 0.1)
--shadow-purple-md: 0 4px 6px -1px rgb(91 33 182 / 0.1)
--shadow-purple-lg: 0 10px 15px -3px rgb(91 33 182 / 0.15)
```

## 📋 Usage Examples

### Basic Layout
```jsx
import styles from './Component.module.css';
import { cn } from '@/styles/utilities/layout';

function MyComponent() {
  return (
    <div className={cn(styles.flex, styles.flex_column, styles.flex_gap_md)}>
      <div className={styles.card}>
        <h2 className={styles.text_lg}>Card Title</h2>
        <p className={styles.text_sm}>Card content</p>
      </div>
    </div>
  );
}
```

### Dynamic Styling
```jsx
import { flex, getSpacing } from '@/styles/utilities/layout';

function DynamicComponent({ direction, gap }) {
  const containerStyle = {
    ...flex({ direction, gap }),
    padding: getSpacing('lg'),
  };
  
  return <div style={containerStyle}>Dynamic layout</div>;
}
```

### Responsive Grid
```jsx
function ResponsiveGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
      {items.map(item => (
        <div key={item.id} className="card card-interactive">
          {item.content}
        </div>
      ))}
    </div>
  );
}
```

## 🛠️ Development Guidelines

### CSS Architecture Principles
1. **Mobile-first**: Start with mobile styles, add larger breakpoints
2. **Component-scoped**: Use CSS Modules for component-specific styles
3. **Utility-first**: Use Tailwind utilities for common patterns
4. **Design tokens**: Always use CSS custom properties, never hardcode values
5. **Performance**: Optimize for bundle size and runtime performance

### Naming Conventions
- **CSS Modules**: `componentName_variant_modifier` (e.g., `card_elevated_interactive`)
- **CSS Custom Properties**: `--prefix-category-property` (e.g., `--color-text-primary`)
- **Utility Classes**: Tailwind conventions (`text-lg`, `p-4`, `bg-primary`)

### Best Practices
- Use semantic HTML elements
- Ensure keyboard accessibility
- Test in both light and dark themes
- Optimize for touch devices
- Follow WCAG 2.1 AA guidelines
- Use progressive enhancement

---

Built with ❤️ for the BWR Plots application using modern CSS architecture and design system principles.