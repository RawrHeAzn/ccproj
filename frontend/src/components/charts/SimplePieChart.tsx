import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface SimplePieChartProps {
  data: unknown[];
  nameKey: string;
  dataKey: string;
}

// Define some colors for the pie slices
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

const SimplePieChart: React.FC<SimplePieChartProps> = ({ data, nameKey, dataKey }) => {
    if (!Array.isArray(data) || data.length === 0) {
        return <p className="text-sm text-gray-500">No data available for chart.</p>;
      }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
        //   label={renderCustomizedLabel} // Optional: Add custom labels
          outerRadius={80}
          fill="#8884d8"
          dataKey={dataKey}
          nameKey={nameKey} // Tooltip will use this
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value: number) => value.toFixed(2)} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
};

export default SimplePieChart;

// Optional: Example for custom labels if needed
/*
const RADIAN = Math.PI / 180;
const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index }) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};
*/ 