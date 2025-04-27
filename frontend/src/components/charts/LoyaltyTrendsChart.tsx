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
import { LoyaltyTrend } from '../../types/dashboardData';

interface LoyaltyTrendsChartProps {
  data: LoyaltyTrend[];
}

// Helper to process data for multi-line chart
const processLoyaltyData = (data: LoyaltyTrend[]) => {
  const processedData: { [key: string]: { name: string; loyalSpend: number; nonLoyalSpend: number } } = {};

  data.forEach(item => {
    const key = `${item.Year}-W${item.Week_num}`;
    if (!processedData[key]) {
      processedData[key] = { name: key, loyalSpend: 0, nonLoyalSpend: 0 };
    }
    if (item.Loyalty_flag === '1') {
        processedData[key].loyalSpend = (processedData[key].loyalSpend || 0) + item.total_spend;
    } else {
        processedData[key].nonLoyalSpend = (processedData[key].nonLoyalSpend || 0) + item.total_spend;
    }
  });

  // Convert to array and sort (optional but good practice)
  return Object.values(processedData).sort((a, b) => {
      // Simple sort by name (Year-Week)
      return a.name.localeCompare(b.name);
  }); 
};


const LoyaltyTrendsChart: React.FC<LoyaltyTrendsChartProps> = ({ data }) => {
  if (!Array.isArray(data) || data.length === 0) {
    return <p className="text-sm text-gray-500">No loyalty trend data available.</p>;
  }

  const chartData = processLoyaltyData(data);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={chartData}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip formatter={(value: number) => value.toFixed(2)} />
        <Legend />
        <Line type="monotone" dataKey="loyalSpend" name="Loyal Customer Spend" stroke="#8884d8" activeDot={{ r: 8 }} />
        <Line type="monotone" dataKey="nonLoyalSpend" name="Non-Loyal Customer Spend" stroke="#82ca9d" activeDot={{ r: 8 }} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LoyaltyTrendsChart; 