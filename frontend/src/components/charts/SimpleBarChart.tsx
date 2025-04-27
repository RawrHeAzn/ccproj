import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Label,
} from 'recharts';

interface SimpleBarChartProps {
  data: unknown[]; // Use unknown[] instead of any[]
  xAxisKey: string;
  barDataKey: string;
  fillColor?: string;
  xAxisLabel?: string; // Optional label for the axis
  yAxisLabel?: string; // Add Y-axis label prop
}

const SimpleBarChart: React.FC<SimpleBarChartProps> = ({ data, xAxisKey, barDataKey, fillColor = "#8884d8", xAxisLabel, yAxisLabel }) => {
  // Check if data is an array before proceeding
  if (!Array.isArray(data) || data.length === 0) { 
    return <p className="text-sm text-gray-500">No data available for chart.</p>;
  }

  // Adjust height to accommodate rotated labels
  const chartHeight = xAxisLabel ? 400 : 350; 

  return (
    <ResponsiveContainer width="100%" height={chartHeight}>
      <BarChart
        data={data} // Pass unknown[] - Recharts will look for keys
        margin={{
          top: 5,
          right: 30,
          left: yAxisLabel ? 30 : 20,
          bottom: xAxisLabel ? 70 : 50,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
            dataKey={xAxisKey} 
            tick={{ fontSize: 10 }} 
            angle={-45} 
            textAnchor="end" 
            interval={0}
        >
            {xAxisLabel && <Label value={xAxisLabel} offset={-55} position="insideBottom" />}
        </XAxis>
        <YAxis tick={{ fontSize: 10 }}>
            {yAxisLabel && <Label value={yAxisLabel} angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />}
        </YAxis>
        <Tooltip formatter={(value: number | string) => 
            typeof value === 'number' ? value.toFixed(2) : value
        } />
        <Legend wrapperStyle={{ paddingTop: '20px' }} />
        <Bar dataKey={barDataKey} fill={fillColor} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default SimpleBarChart; 