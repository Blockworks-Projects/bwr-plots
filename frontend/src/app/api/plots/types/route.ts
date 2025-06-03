import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  try {
    const plotTypes = [
      {
        type: 'line',
        name: 'Line Chart',
        description: 'Best for showing trends over time',
        required_columns: ['x', 'y'],
        optional_columns: ['color'],
        supports_time_series: true,
        icon: '📈'
      },
      {
        type: 'bar',
        name: 'Bar Chart', 
        description: 'Best for comparing categories',
        required_columns: ['x', 'y'],
        optional_columns: ['color'],
        supports_time_series: false,
        icon: '📊'
      },
      {
        type: 'scatter',
        name: 'Scatter Plot',
        description: 'Best for showing relationships between variables',
        required_columns: ['x', 'y'],
        optional_columns: ['color', 'size'],
        supports_time_series: false,
        icon: '⚬'
      },
      {
        type: 'area',
        name: 'Area Chart',
        description: 'Best for showing cumulative data over time',
        required_columns: ['x', 'y'],
        optional_columns: ['color'],
        supports_time_series: true,
        icon: '🏔️'
      },
      {
        type: 'histogram',
        name: 'Histogram',
        description: 'Best for showing distribution of a single variable',
        required_columns: ['x'],
        optional_columns: ['color'],
        supports_time_series: false,
        icon: '📊'
      },
      {
        type: 'box',
        name: 'Box Plot',
        description: 'Best for showing statistical distribution',
        required_columns: ['y'],
        optional_columns: ['x', 'color'],
        supports_time_series: false,
        icon: '📦'
      },
      {
        type: 'heatmap',
        name: 'Heatmap',
        description: 'Best for showing relationships in 2D data',
        required_columns: ['x', 'y', 'z'],
        optional_columns: [],
        supports_time_series: false,
        icon: '🔥'
      }
    ];

    return NextResponse.json({
      plot_types: plotTypes,
      total_count: plotTypes.length
    });
    
  } catch (error) {
    console.error('Plot types error:', error);
    return NextResponse.json({ 
      error: 'Failed to get plot types' 
    }, { status: 500 });
  }
} 