import React from 'react';
import { FrequentPair } from '../../types/dashboardData'; // Use defined type

interface FrequentPairsTableProps {
  data: FrequentPair[];
}

const FrequentPairsTable: React.FC<FrequentPairsTableProps> = ({ data }) => {
  if (!Array.isArray(data) || data.length === 0) {
    return <p className="text-sm text-gray-500">No frequent pair data available.</p>;
  }

  return (
    <div className="overflow-x-auto shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-300">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">Item 1 (Commodity)</th>
                <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Item 2 (Commodity)</th>
                <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6 text-right text-sm font-semibold text-gray-900">Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {data.map((item, index) => (
                <tr key={`${item.item1}-${item.item2}-${index}`}> 
                  <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">{item.item1}</td>
                  <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{item.item2}</td>
                  <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">{item.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
    </div>
  );
};

export default FrequentPairsTable; 