// Dummy data generators for each plot type

export type DummyDataType = 
  | 'scatter'
  | 'metric_share_area'
  | 'bar'
  | 'horizontal_bar'
  | 'multi_bar'
  | 'stacked_bar'
  | 'table';

export interface DummyDataset {
  id: string;
  type: DummyDataType;
  name: string;
  description: string;
  data: any[];
  columns: string[];
}

// Generate dates for time series
const generateDates = (days: number): string[] => {
  const dates: string[] = [];
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);
  
  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    dates.push(date.toISOString().split('T')[0]);
  }
  return dates;
};

// Random walk for price-like data
const randomWalk = (start: number, steps: number, volatility: number = 0.02): number[] => {
  const values = [start];
  for (let i = 1; i < steps; i++) {
    const change = (Math.random() - 0.5) * 2 * volatility;
    values.push(Math.max(0, values[i - 1] * (1 + change)));
  }
  return values;
};

export const dummyDatasets: DummyDataset[] = [
  {
    id: 'crypto-price-comparison',
    type: 'scatter',
    name: 'Crypto Price Comparison',
    description: 'BTC vs ETH daily prices over 90 days',
    columns: ['date', 'BTC_Price', 'ETH_Price'],
    data: (() => {
      const dates = generateDates(90);
      const btcPrices = randomWalk(45000, 90, 0.015);
      const ethPrices = randomWalk(2500, 90, 0.02);
      
      return dates.map((date, i) => ({
        date,
        BTC_Price: Math.round(btcPrices[i]),
        ETH_Price: Math.round(ethPrices[i])
      }));
    })()
  },
  
  {
    id: 'defi-market-share',
    type: 'metric_share_area',
    name: 'DeFi Protocol Market Share',
    description: 'Market share of top DeFi protocols over time',
    columns: ['date', 'Uniswap', 'Aave', 'Compound', 'MakerDAO', 'Curve'],
    data: (() => {
      const dates = generateDates(120);
      return dates.map((date) => {
        const total = 100;
        const uniswap = 25 + Math.random() * 10;
        const aave = 20 + Math.random() * 8;
        const compound = 15 + Math.random() * 5;
        const maker = 18 + Math.random() * 7;
        const curve = total - uniswap - aave - compound - maker;
        
        return {
          date,
          Uniswap: Math.round(uniswap),
          Aave: Math.round(aave),
          Compound: Math.round(compound),
          MakerDAO: Math.round(maker),
          Curve: Math.round(curve)
        };
      });
    })()
  },
  
  {
    id: 'monthly-trading-volume',
    type: 'bar',
    name: 'Monthly Trading Volume',
    description: 'Trading volume by month',
    columns: ['month', 'volume'],
    data: [
      { month: 'January', volume: 1250000000 },
      { month: 'February', volume: 1450000000 },
      { month: 'March', volume: 1650000000 },
      { month: 'April', volume: 1350000000 },
      { month: 'May', volume: 1850000000 },
      { month: 'June', volume: 2050000000 }
    ]
  },
  
  {
    id: 'top-tokens-market-cap',
    type: 'horizontal_bar',
    name: 'Top 10 Tokens by Market Cap',
    description: 'Horizontal bar chart of token market caps',
    columns: ['token', 'market_cap'],
    data: [
      { token: 'Bitcoin', market_cap: 850000000000 },
      { token: 'Ethereum', market_cap: 380000000000 },
      { token: 'Tether', market_cap: 95000000000 },
      { token: 'BNB', market_cap: 85000000000 },
      { token: 'XRP', market_cap: 65000000000 },
      { token: 'Solana', market_cap: 60000000000 },
      { token: 'USDC', market_cap: 45000000000 },
      { token: 'Cardano', market_cap: 35000000000 },
      { token: 'Avalanche', market_cap: 25000000000 },
      { token: 'Dogecoin', market_cap: 20000000000 }
    ]
  },
  
  {
    id: 'quarterly-revenue',
    type: 'multi_bar',
    name: 'Quarterly Revenue by Product',
    description: 'Revenue comparison across products by quarter',
    columns: ['quarter', 'Trading_Fees', 'Staking_Revenue', 'Lending_Income'],
    data: [
      { quarter: 'Q1 2023', Trading_Fees: 45000000, Staking_Revenue: 23000000, Lending_Income: 18000000 },
      { quarter: 'Q2 2023', Trading_Fees: 52000000, Staking_Revenue: 28000000, Lending_Income: 22000000 },
      { quarter: 'Q3 2023', Trading_Fees: 48000000, Staking_Revenue: 31000000, Lending_Income: 25000000 },
      { quarter: 'Q4 2023', Trading_Fees: 58000000, Staking_Revenue: 35000000, Lending_Income: 28000000 },
      { quarter: 'Q1 2024', Trading_Fees: 62000000, Staking_Revenue: 38000000, Lending_Income: 32000000 }
    ]
  },
  
  {
    id: 'protocol-tvl-composition',
    type: 'stacked_bar',
    name: 'Protocol TVL Composition',
    description: 'Total Value Locked breakdown by asset type',
    columns: ['protocol', 'ETH', 'USDC', 'WBTC', 'Other'],
    data: [
      { protocol: 'Aave', ETH: 2500000000, USDC: 1800000000, WBTC: 800000000, Other: 400000000 },
      { protocol: 'Compound', ETH: 1800000000, USDC: 2200000000, WBTC: 600000000, Other: 300000000 },
      { protocol: 'MakerDAO', ETH: 3200000000, USDC: 1500000000, WBTC: 500000000, Other: 800000000 },
      { protocol: 'Curve', ETH: 1200000000, USDC: 3500000000, WBTC: 300000000, Other: 1000000000 },
      { protocol: 'Uniswap', ETH: 2800000000, USDC: 2000000000, WBTC: 700000000, Other: 500000000 }
    ]
  },
  
  {
    id: 'token-performance-metrics',
    type: 'table',
    name: 'Token Performance Metrics',
    description: 'Comprehensive token performance table',
    columns: ['Token', 'Price', '24h_Change', '7d_Change', 'Volume', 'Market_Cap'],
    data: [
      { Token: 'BTC', Price: '$45,234', '24h_Change': '+2.3%', '7d_Change': '+5.8%', Volume: '$28.5B', Market_Cap: '$885B' },
      { Token: 'ETH', Price: '$2,567', '24h_Change': '+3.1%', '7d_Change': '+8.2%', Volume: '$15.2B', Market_Cap: '$308B' },
      { Token: 'SOL', Price: '$98.45', '24h_Change': '-1.2%', '7d_Change': '+12.5%', Volume: '$2.8B', Market_Cap: '$42B' },
      { Token: 'AVAX', Price: '$35.67', '24h_Change': '+4.5%', '7d_Change': '+15.3%', Volume: '$890M', Market_Cap: '$13B' },
      { Token: 'MATIC', Price: '$0.92', '24h_Change': '+1.8%', '7d_Change': '+6.7%', Volume: '$650M', Market_Cap: '$8.5B' }
    ]
  }
];

export function getDummyDataByType(type: DummyDataType): DummyDataset | undefined {
  return dummyDatasets.find(dataset => dataset.type === type);
}

export function getAllDummyDatasets(): DummyDataset[] {
  return dummyDatasets;
}

// Convert dataset to CSV format
export function datasetToCSV(dataset: DummyDataset): string {
  const headers = dataset.columns.join(',');
  const rows = dataset.data.map(row => 
    dataset.columns.map(col => {
      const value = row[col];
      // Quote strings that contain commas, newlines, or quotes
      if (typeof value === 'string' && (value.includes(',') || value.includes('\n') || value.includes('"'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return value;
    }).join(',')
  );
  
  return [headers, ...rows].join('\n');
}

// Create a CSV file from dataset
export function createCSVFile(dataset: DummyDataset): File {
  const csvContent = datasetToCSV(dataset);
  const blob = new Blob([csvContent], { type: 'text/csv' });
  return new File([blob], `${dataset.name.replace(/\s+/g, '_')}.csv`, { type: 'text/csv' });
}