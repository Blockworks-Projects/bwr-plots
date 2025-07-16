// Simple plot generator using Plotly
import { DummyDataType } from './dummyData';

export function generatePlotHtml(
  plotType: string,
  data: any[],
  config: any
): string {
  const title = config.title || 'Chart';
  const subtitle = config.subtitle || '';
  
  // Create a simple Plotly chart based on type
  let plotlyData: any[] = [];
  let layout: any = {
    title: {
      text: title + (subtitle ? `<br><sub>${subtitle}</sub>` : ''),
      font: { size: 24 }
    },
    margin: { t: 100, b: 50, l: 50, r: 50 },
    plot_bgcolor: '#1a1a1a',
    paper_bgcolor: '#0a0a0a',
    font: { color: '#e5e5e5' },
    xaxis: { 
      gridcolor: '#333',
      title: config.axis_config?.x_title || ''
    },
    yaxis: { 
      gridcolor: '#333',
      title: config.axis_config?.y_title || ''
    }
  };

  switch (plotType) {
    case 'scatter':
    case 'line':
      // Get the first two numeric columns
      const xCol = config.x_column || Object.keys(data[0])[0];
      const yCol = config.y_column || Object.keys(data[0])[1];
      
      plotlyData = [{
        x: data.map(row => row[xCol]),
        y: data.map(row => row[yCol]),
        type: 'scatter',
        mode: 'lines+markers',
        name: yCol,
        line: { color: '#4a90e2', width: 2 },
        marker: { color: '#4a90e2', size: 6 }
      }];
      break;

    case 'bar':
      const barX = config.x_column || Object.keys(data[0])[0];
      const barY = config.y_column || Object.keys(data[0])[1];
      
      plotlyData = [{
        x: data.map(row => row[barX]),
        y: data.map(row => row[barY]),
        type: 'bar',
        marker: { color: '#4a90e2' }
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
        marker: { color: '#4a90e2' }
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
        marker: { color: ['#4a90e2', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][i % 5] }
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
        fillcolor: ['#4a90e2', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][i % 5]
      }));
      break;

    case 'table':
      const headers = Object.keys(data[0]);
      const values = headers.map(header => data.map(row => row[header]));
      
      plotlyData = [{
        type: 'table',
        header: {
          values: headers,
          align: 'center',
          line: { width: 1, color: '#333' },
          fill: { color: '#1a1a1a' },
          font: { color: '#e5e5e5', size: 12 }
        },
        cells: {
          values: values,
          align: 'center',
          line: { color: '#333', width: 1 },
          fill: { color: '#0a0a0a' },
          font: { color: '#e5e5e5', size: 11 }
        }
      }];
      
      layout = { ...layout, height: 600 };
      break;

    default:
      // Default to scatter plot
      plotlyData = [{
        x: data.map((_, i) => i),
        y: data.map(() => Math.random() * 100),
        type: 'scatter',
        mode: 'lines+markers'
      }];
  }

  // Generate the HTML with Plotly
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <style>
        body { margin: 0; padding: 0; background: #0a0a0a; }
        #plotly-div { width: 100%; height: 100%; }
      </style>
    </head>
    <body>
      <div id="plotly-div"></div>
      <script>
        const data = ${JSON.stringify(plotlyData)};
        const layout = ${JSON.stringify(layout)};
        const config = { responsive: true, displayModeBar: true };
        
        console.log('Plotting with data:', data);
        console.log('Layout:', layout);
        
        Plotly.newPlot('plotly-div', data, layout, config);
      </script>
    </body>
    </html>
  `;

  return html;
}