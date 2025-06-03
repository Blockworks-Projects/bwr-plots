import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  try {
    const watermarks = [
      {
        id: 'bwr',
        name: 'Blockworks Research',
        description: 'Standard BWR watermark',
        default: true
      },
      {
        id: 'blockworks',
        name: 'Blockworks',
        description: 'Blockworks company watermark',
        default: false
      },
      {
        id: 'none',
        name: 'No Watermark',
        description: 'No watermark applied',
        default: false
      },
      {
        id: 'custom',
        name: 'Custom',
        description: 'Custom watermark text',
        default: false,
        supports_custom_text: true
      }
    ];

    const colorSchemes = [
      {
        id: 'blockworks',
        name: 'Blockworks',
        description: 'Official Blockworks color scheme',
        default: true,
        colors: ['#FF6B35', '#F7931A', '#004E89', '#1A659E', '#65AFFF']
      },
      {
        id: 'viridis',
        name: 'Viridis',
        description: 'Perceptually uniform color scheme',
        default: false,
        colors: ['#440154', '#482777', '#3F4A8A', '#31678E', '#26838F']
      },
      {
        id: 'plasma',
        name: 'Plasma',
        description: 'High contrast plasma colors',
        default: false,
        colors: ['#0C0887', '#5B02A3', '#9A179B', '#CB4678', '#F0F921']
      },
      {
        id: 'grayscale',
        name: 'Grayscale',
        description: 'Monochrome color scheme',
        default: false,
        colors: ['#000000', '#404040', '#808080', '#C0C0C0', '#FFFFFF']
      }
    ];

    return NextResponse.json({
      watermarks,
      color_schemes: colorSchemes,
      default_watermark: 'bwr',
      default_color_scheme: 'blockworks'
    });
    
  } catch (error) {
    console.error('Watermarks config error:', error);
    return NextResponse.json({ 
      error: 'Failed to get watermark configuration' 
    }, { status: 500 });
  }
} 