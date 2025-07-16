// BWR-styled plot generator that matches the Python library
import { DummyDataType } from './dummyData';

// BWR Colors
const BWR_COLORS = {
  background: '#1A1A1A',
  primary: '#5637cd',
  default_palette: [
    '#5637cd',
    '#779BE7', 
    '#8F7BE1',
    '#EF798A',
    '#C0B9D8',
    '#8a7cff',
    '#F3A712',
    '#9f95c6',
    '#d62728',
    '#9467bd'
  ],
  text: '#ededed',
  subtitle: '#adb0b5',
  annotation: '#9f95c6'
};

// BWR Fonts
const BWR_FONTS = {
  family: 'Maison Neue, Inter, sans-serif',
  title_size: 52,
  subtitle_size: 22,
  axis_size: 17,
  tick_size: 22,
  legend_size: 24
};

export function generateBWRPlotHtml(
  plotType: string,
  data: any[],
  config: any
): string {
  const title = config.title || 'BWR Chart';
  const subtitle = config.subtitle || '';
  const source = config.source || '';
  
  // Create BWR-styled layout
  let layout: any = {
    title: {
      text: title + (subtitle ? `<br><sub style="font-size: ${BWR_FONTS.subtitle_size}px; color: ${BWR_COLORS.subtitle};">${subtitle}</sub>` : ''),
      font: { 
        family: BWR_FONTS.family,
        size: BWR_FONTS.title_size, 
        color: BWR_COLORS.text 
      },
      x: 0.035,
      y: 0.95,
      xanchor: 'left'
    },
    paper_bgcolor: BWR_COLORS.background,
    plot_bgcolor: BWR_COLORS.background,
    font: { 
      family: BWR_FONTS.family,
      color: BWR_COLORS.text,
      size: BWR_FONTS.tick_size 
    },
    margin: { 
      l: 120, 
      r: 70, 
      t: 150, 
      b: source ? 120 : 80 
    },
    hovermode: 'x unified',
    hoverdistance: 100,
    xaxis: {
      title: {
        text: config.axis_config?.x_title || '',
        font: { 
          family: BWR_FONTS.family,
          size: BWR_FONTS.axis_size, 
          color: BWR_COLORS.text 
        }
      },
      tickfont: { 
        family: BWR_FONTS.family,
        size: BWR_FONTS.tick_size, 
        color: BWR_COLORS.text 
      },
      gridcolor: '#333333',
      linecolor: '#555555',
      zerolinecolor: '#555555'
    },
    yaxis: {
      title: {
        text: config.axis_config?.y_title || '',
        font: { 
          family: BWR_FONTS.family,
          size: BWR_FONTS.axis_size, 
          color: BWR_COLORS.text 
        }
      },
      tickfont: { 
        family: BWR_FONTS.family,
        size: BWR_FONTS.tick_size, 
        color: BWR_COLORS.text 
      },
      gridcolor: '#333333',
      linecolor: '#555555',
      zerolinecolor: '#555555'
    },
    legend: {
      font: { 
        family: BWR_FONTS.family,
        size: BWR_FONTS.legend_size, 
        color: BWR_COLORS.text 
      },
      bgcolor: 'rgba(255,255,255,0)',
      bordercolor: 'rgba(255,255,255,0)',
      orientation: 'h',
      yanchor: 'top',
      y: -0.15,
      xanchor: 'left',
      x: 0
    }
  };

  // Add source annotation if provided
  if (source) {
    layout.annotations = [{
      text: source,
      x: 1.002,
      y: -0.16,
      xref: 'paper',
      yref: 'paper',
      xanchor: 'right',
      yanchor: 'top',
      showarrow: false,
      font: {
        family: BWR_FONTS.family,
        size: BWR_FONTS.axis_size,
        color: BWR_COLORS.annotation
      }
    }];
  }

  let plotlyData: any[] = [];

  switch (plotType) {
    case 'scatter':
    case 'line':
      const xCol = config.x_column || Object.keys(data[0])[0];
      const yCol = config.y_column || Object.keys(data[0])[1];
      
      plotlyData = [{
        x: data.map(row => row[xCol]),
        y: data.map(row => row[yCol]),
        type: 'scatter',
        mode: 'lines+markers',
        name: yCol,
        line: { 
          color: BWR_COLORS.primary, 
          width: 3 
        },
        marker: { 
          color: BWR_COLORS.primary, 
          size: 6,
          line: { width: 0 }
        }
      }];
      break;

    case 'bar':
      const barX = config.x_column || Object.keys(data[0])[0];
      const barY = config.y_column || Object.keys(data[0])[1];
      
      plotlyData = [{
        x: data.map(row => row[barX]),
        y: data.map(row => row[barY]),
        type: 'bar',
        marker: { 
          color: BWR_COLORS.primary,
          line: { width: 0 }
        }
      }];
      break;

    case 'horizontal_bar':
      const hbarCat = config.x_column || Object.keys(data[0])[0];
      const hbarVal = config.y_column || Object.keys(data[0])[1];
      
      plotlyData = [{
        x: data.map(row => row[hbarVal]),
        y: data.map(row => row[hbarCat]),
        type: 'bar',
        orientation: 'h',
        marker: { 
          color: BWR_COLORS.primary,
          line: { width: 0 }
        }
      }];
      break;

    case 'multi_bar':
    case 'stacked_bar':
      const groupCol = config.x_column || Object.keys(data[0])[0];
      const valueColumns = Object.keys(data[0]).filter(col => col !== groupCol);
      
      plotlyData = valueColumns.map((col, i) => ({
        x: data.map(row => row[groupCol]),
        y: data.map(row => row[col]),
        type: 'bar',
        name: col,
        marker: { 
          color: BWR_COLORS.default_palette[i % BWR_COLORS.default_palette.length],
          line: { width: 0 }
        }
      }));
      
      if (plotType === 'stacked_bar') {
        layout.barmode = 'stack';
      } else {
        layout.barmode = 'group';
      }
      break;

    case 'metric_share_area':
      const dateCol = config.x_column || Object.keys(data[0])[0];
      const metricColumns = Object.keys(data[0]).filter(col => col !== dateCol);
      
      plotlyData = metricColumns.map((col, i) => ({
        x: data.map(row => row[dateCol]),
        y: data.map(row => row[col]),
        type: 'scatter',
        mode: 'lines',
        stackgroup: 'one',
        groupnorm: 'percent',
        name: col,
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: BWR_COLORS.default_palette[i % BWR_COLORS.default_palette.length]
      }));
      
      // Update Y-axis to show percentages
      layout.yaxis.tickformat = '.1%';
      layout.yaxis.title.text = layout.yaxis.title.text || 'Percentage';
      break;

    case 'table':
      const headers = Object.keys(data[0]);
      const values = headers.map(header => data.map(row => row[header]));
      
      plotlyData = [{
        type: 'table',
        header: {
          values: headers,
          align: 'center',
          line: { width: 1, color: '#555555' },
          fill: { color: '#2a2a2a' },
          font: { 
            family: BWR_FONTS.family,
            color: BWR_COLORS.text, 
            size: 24 
          }
        },
        cells: {
          values: values,
          align: 'center',
          line: { color: '#555555', width: 1 },
          fill: { color: BWR_COLORS.background },
          font: { 
            family: BWR_FONTS.family,
            color: BWR_COLORS.text, 
            size: 20 
          }
        }
      }];
      
      layout = { 
        ...layout, 
        height: 600,
        margin: { l: 10, r: 10, t: 100, b: 10 }
      };
      break;

    default:
      // Default to scatter plot
      plotlyData = [{
        x: data.map((_, i) => i),
        y: data.map(() => Math.random() * 100),
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: BWR_COLORS.primary }
      }];
  }

  // Generate the HTML with BWR styling
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
      <style>
        body { 
          margin: 0; 
          padding: 0; 
          background: ${BWR_COLORS.background}; 
          font-family: 'Inter', 'Maison Neue', sans-serif;
        }
        #plotly-div { 
          width: 100%; 
          height: 100vh; 
          background: ${BWR_COLORS.background};
        }
      </style>
    </head>
    <body>
      <div id="plotly-div"></div>
      <script>
        const data = ${JSON.stringify(plotlyData)};
        const layout = ${JSON.stringify(layout)};
        const config = { 
          responsive: true, 
          displayModeBar: true,
          modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
          displaylogo: false,
          toImageButtonOptions: {
            format: 'png',
            filename: 'bwr_chart',
            height: 1080,
            width: 1920,
            scale: 1
          }
        };
        
        console.log('BWR Plot - Data:', data);
        console.log('BWR Plot - Layout:', layout);
        
        Plotly.newPlot('plotly-div', data, layout, config);
      </script>
    </body>
    </html>
  `;

  return html;
}