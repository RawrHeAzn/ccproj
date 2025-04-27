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
} from 'recharts';

interface SimpleBarChartProps {
  data: unknown[]; // Use unknown[] instead of any[]
  xAxisKey: string;
  barDataKey: string;
  fillColor?: string;
}

const SimpleBarChart: React.FC<SimpleBarChartProps> = ({ data, xAxisKey, barDataKey, fillColor = "#8884d8" }) => {
  // Check if data is an array before proceeding
  if (!Array.isArray(data) || data.length === 0) { 
    return <p className="text-sm text-gray-500">No data available for chart.</p>;
  }

  // Recharts can often handle data with appropriate keys even if type is unknown[]
  // Add more specific checks here if needed for complex scenarios
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={data} // Pass unknown[] - Recharts will look for keys
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip formatter={(value: number) => value.toFixed(2)} />
        <Legend />
        <Bar dataKey={barDataKey} fill={fillColor} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default SimpleBarChart; 