import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface SimpleLineChartProps {
  data: unknown[];
  xAxisKey: string;
  lineDataKey: string;
  strokeColor?: string;
}

const SimpleLineChart: React.FC<SimpleLineChartProps> = ({ data, xAxisKey, lineDataKey, strokeColor = "#8884d8" }) => {
  if (!Array.isArray(data) || data.length === 0) {
    return <p className="text-sm text-gray-500">No data available for chart.</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={data}
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
        <Line type="monotone" dataKey={lineDataKey} stroke={strokeColor} activeDot={{ r: 8 }} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default SimpleLineChart; 