import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  try {
    const plotTypes = [
      {
        type: 'scatter',
        name: 'Scatter/Line Plot',
        description: 'Show relationships between variables or trends over time',
        required_columns: ['x', 'y'],
        optional_columns: ['color', 'size'],
        supports_time_series: true,
        icon: '📈'
      },
      {
        type: 'metric_share_area',
        name: 'Metric Share Area',
        description: 'Show percentage composition over time',
        required_columns: ['date'],
        optional_columns: [],
        supports_time_series: true,
        icon: '📊'
      },
      {
        type: 'bar',
        name: 'Bar Chart', 
        description: 'Compare values across categories',
        required_columns: ['x', 'y'],
        optional_columns: ['color'],
        supports_time_series: false,
        icon: '📊'
      },
      {
        type: 'horizontal_bar',
        name: 'Horizontal Bar Chart',
        description: 'Compare values with horizontal bars - great for long labels',
        required_columns: ['category', 'value'],
        optional_columns: ['color'],
        supports_time_series: false,
        icon: '📊'
      },
      {
        type: 'multi_bar',
        name: 'Multi-Bar Chart',
        description: 'Compare multiple series across categories',
        required_columns: ['x'],
        optional_columns: ['color'],
        supports_time_series: true,
        icon: '📊'
      },
      {
        type: 'stacked_bar',
        name: 'Stacked Bar Chart',
        description: 'Show composition of totals across categories',
        required_columns: ['x'],
        optional_columns: [],
        supports_time_series: true,
        icon: '📊'
      },
      {
        type: 'table',
        name: 'Data Table',
        description: 'Display data in a formatted table',
        required_columns: [],
        optional_columns: [],
        supports_time_series: false,
        icon: '📋'
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