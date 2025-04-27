import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext'; // To get the token

// Define the structure of a search result item
interface SearchResult {
  Hshd_num: number;
  Basket_num: number;
  Date: string; // Assuming date comes as string, format later if needed
  Product_num: number;
  Department: string;
  Commodity: string;
  Spend: number;
  Units: number;
}

const SearchPage: React.FC = () => {
  const [hshdNum, setHshdNum] = useState<string>('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false); // Track if a search has been performed

  const { token } = useAuth(); // Get the auth token
  const API_BASE_URL = 'https://cloud-comp-retail.vercel.app';

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hshdNum.trim()) {
      setError('Please enter a Household Number.');
      return;
    }
    if (!token) {
        setError('Authentication token not found. Please log in again.');
        return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setSearched(true);

    try {
      const response = await fetch(`${API_BASE_URL}/household-search/${hshdNum}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            // Include the token in the Authorization header
            // Adjust if your backend expects a different scheme or header name
            'Authorization': `Bearer ${token}` 
        },
      });
      
      if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data: SearchResult[] = await response.json();
      setResults(data);

    } catch (err) {
        if (err instanceof Error) {
            setError(err.message || 'An error occurred during search.');
        } else {
            setError('An unknown error occurred during search.');
        }
        setResults([]); // Clear results on error
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6 text-indigo-800 border-b pb-2 border-indigo-200">Search Transactions</h2>
      
      <form onSubmit={handleSearch} className="mb-8 flex items-center space-x-4 bg-gray-50 p-4 rounded-lg border border-gray-200 shadow-sm">
        <input
          type="number"
          value={hshdNum}
          onChange={(e) => setHshdNum(e.target.value)}
          placeholder="Enter Household Number"
          required
          className="block w-full max-w-xs px-4 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent sm:text-sm"
        />
        <button
          type="submit"
          disabled={loading}
          className="inline-flex justify-center py-2 px-5 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-150 ease-in-out"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && (
        <p className="text-sm text-red-600 mb-4">Error: {error}</p>
      )}

      {loading && (
        <p className="text-sm text-gray-600">Loading results...</p>
      )}

      {!loading && searched && results.length === 0 && !error && (
         <p className="text-sm text-gray-600">No transactions found for this household number.</p>
      )}

      {!loading && results.length > 0 && (
        <div className="overflow-hidden shadow-md ring-1 ring-black ring-opacity-5 rounded-lg">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-300">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">Hshd Num</th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Basket Num</th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Date</th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Product Num</th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Department</th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Commodity</th>
                  <th scope="col" className="px-3 py-3.5 text-right text-sm font-semibold text-gray-900">Spend</th>
                  <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6 text-right text-sm font-semibold text-gray-900">Units</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {results.map((item, index) => (
                  <tr key={`${item.Basket_num}-${item.Product_num}-${index}`}> {/* Composite key */} 
                    <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">{item.Hshd_num}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{item.Basket_num}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{item.Date}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{item.Product_num}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{item.Department}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{item.Commodity}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500 text-right">{item.Spend.toFixed(2)}</td>
                    <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">{item.Units}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchPage; 