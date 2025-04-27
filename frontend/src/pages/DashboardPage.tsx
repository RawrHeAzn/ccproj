import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import {
    TopSpender,
    LoyaltyTrend,
    EngagementByIncome,
    BrandPreference,
    FrequentPair,
    PopularProduct,
    SeasonalTrend,
    ChurnRiskCustomer,
    ChurnRiskData,
    SummaryCount,
} from '../types/dashboardData';

// Import Chart Components
import SimpleBarChart from '../components/charts/SimpleBarChart';
import SimpleLineChart from '../components/charts/SimpleLineChart';
import SimplePieChart from '../components/charts/SimplePieChart';
import FrequentPairsTable from '../components/charts/FrequentPairsTable';
import LoyaltyTrendsChart from '../components/charts/LoyaltyTrendsChart';
// Import react-select
import Select, { MultiValue, StylesConfig } from 'react-select';
import { ResponsiveContainer, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

// Type for API fetching state
interface FetchState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  isCalculating: boolean; // NEW state to track 503 status
}

// Custom hook for fetching data with retry on 503
function useFetchDashboardData<T>(endpoint: string, retryDelay = 5000, maxRetries = 12): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    loading: true,
    error: null,
    isCalculating: false, // Initialize
  });
  const { token } = useAuth();
  const API_BASE_URL = 'https://ccproj.onrender.com'.replace(/\/$/, ''); 
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    if (!token) {
      setState({ data: null, loading: false, error: 'Authentication token not found.', isCalculating: false });
      return;
    }

    let isMounted = true; // Track if component is still mounted
    let timeoutId: NodeJS.Timeout | null = null; // Track retry timeout

    const fetchData = async () => {
      // Don't reset loading to true on retries, keep showing calculating state
      if (retryCount === 0) {
           setState(prev => ({ ...prev, loading: true, error: null, isCalculating: false }));
      }
      
      // Ensure endpoint doesn't start with /
      const cleanEndpoint = endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
      // Construct URL robustly
      const fullUrl = `${API_BASE_URL}/${cleanEndpoint}`;
      
      // Use console.log for frontend logging
      console.log(`Fetching from: ${fullUrl} (Attempt: ${retryCount + 1})`); 

      try {
        const response = await fetch(fullUrl, {
          headers: { 'Authorization': `Bearer ${token}` },
        });

        // --- Handle 503: Data Not Ready --- 
        if (response.status === 503) {
            console.log(`Received 503 for ${endpoint}, will retry...`);
            if (isMounted && retryCount < maxRetries) {
                setState(prev => ({ ...prev, loading: false, error: null, isCalculating: true })); 
                // Schedule retry
                timeoutId = setTimeout(() => setRetryCount(prev => prev + 1), retryDelay);
            } else if (isMounted) {
                 // Max retries reached
                 setState(prev => ({ ...prev, loading: false, error: `Data for ${endpoint} did not become available after ${maxRetries} retries.`, isCalculating: false }));
            }
            return; // Stop processing this fetch attempt
        }

        // --- Handle other errors --- 
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        // --- Handle Success (200 OK) --- 
        const jsonData = await response.json();
        if (isMounted) {
            setState({ data: jsonData, loading: false, error: null, isCalculating: false });
            // No need to retry further
        }

      } catch (err) {
        // --- Handle Fetch/Network Errors --- 
        let errorMessage = 'An unknown error occurred.';
        if (err instanceof Error) {
          errorMessage = err.message;
        }
        if (isMounted) {
             setState({ data: null, loading: false, error: `Failed to fetch ${endpoint}: ${errorMessage}`, isCalculating: false });
             // Stop retrying on non-503 errors
        }
      }
    };

    // Initial fetch or retry fetch
    fetchData();

    // Cleanup function
    return () => {
      isMounted = false; // Mark component as unmounted
      if (timeoutId) {
          clearTimeout(timeoutId); // Clear pending retry timeout
      }
    };
    // Dependencies: trigger fetch on token change, endpoint change, or retry count change
  }, [token, endpoint, retryCount, retryDelay, maxRetries]);

  return state;
}

// Wrapper to handle loading/error states before rendering chart
const ChartWrapper: React.FC<{ title: string; fetchState: FetchState<unknown>; children: React.ReactNode }> = ({ title, fetchState, children }) => {
  const { loading, error, isCalculating } = fetchState; // Use isCalculating state
  return (
    <div className="mb-6 p-5 border rounded-lg bg-gradient-to-br from-white to-gray-50 shadow-md min-h-[200px]">
      <h3 className="text-lg font-semibold mb-3 text-gray-700">{title}</h3>
      {/* Show loading on initial load */}
      {loading && (
        <div className="flex items-center justify-center h-40">
          <p className="text-sm text-gray-500">Loading...</p>
        </div>
      )}
      {/* Show calculating message if retrying after 503 */}
      {isCalculating && !loading && (
          <div className="flex items-center justify-center h-40">
              <p className="text-sm text-blue-600 animate-pulse">Calculating data, please wait...</p>
          </div>
      )}
      {/* Show error only if not loading or calculating */}
      {error && !loading && !isCalculating && (
        <div className="flex items-center justify-center h-40 p-4">
          <p className="text-sm text-red-600 text-center">Error: {error}</p>
        </div>
      )}
      {/* Show children only if data is loaded (not loading, not calculating, no error) */}
      {!loading && !error && !isCalculating && children}
    </div>
  );
};

const DashboardPage: React.FC = () => {
  // Fetch data for charts
  const topSpenders = useFetchDashboardData<TopSpender[]>('/top-spenders');
  const loyaltyTrends = useFetchDashboardData<LoyaltyTrend[]>('/loyalty-trends');
  const engagementByIncome = useFetchDashboardData<EngagementByIncome[]>('/engagement-by-income');
  const brandPreference = useFetchDashboardData<BrandPreference[]>('/brand-preference-split');
  const frequentPairs = useFetchDashboardData<FrequentPair[]>('/frequent-pairs');
  const popularProducts = useFetchDashboardData<PopularProduct[]>('/popular-products');
  const seasonalTrends = useFetchDashboardData<SeasonalTrend[]>('/seasonal-trends');
  const churnRisk = useFetchDashboardData<ChurnRiskData>('/churn-risk');

  // Format seasonal data
  const formatSeasonalData = (data: SeasonalTrend[] | null) => {
    if (!data) return [];
    return data.map(item => ({
      ...item,
      monthLabel: `${item.year}-${String(item.month).padStart(2, '0')}`,
    })).sort((a, b) => a.monthLabel.localeCompare(b.monthLabel));
  };

  // --- ML Prediction States ---
  const [incomeRange, setIncomeRange] = useState('');
  const [hhSize, setHhSize] = useState('');
  const [children, setChildren] = useState('');
  const [predictedCLV, setPredictedCLV] = useState<number | null>(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  
  // --- Basket Prediction States ---
  const [availableFeatures, setAvailableFeatures] = useState<{ value: string, label: string }[]>([]);
  const [selectedCommodities, setSelectedCommodities] = useState<{ value: string, label: string }[]>([]);
  const [basketPredictResult, setBasketPredictResult] = useState<{ target_item: string, probability: number } | null>(null);
  const [basketPredictLoading, setBasketPredictLoading] = useState(false);
  const [basketPredictError, setBasketPredictError] = useState<string | null>(null);
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const { token } = useAuth(); // Get token for API calls
  const API_BASE_URL = 'https://ccproj.onrender.com'; // Updated URL in component scope
  
  // Fetch available features for the basket predictor
  useEffect(() => {
      const fetchFeatures = async () => {
          if (!token) {
              setFeaturesLoading(false);
              setBasketPredictError("Authentication token not found."); // Set error if no token
              return;
          }
          setFeaturesLoading(true);
          try {
              const response = await fetch(`${API_BASE_URL}/get-prediction-features`, {
                  headers: { 'Authorization': `Bearer ${token}` },
              });
              if (!response.ok) {
                  const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
                  throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
              }
              const data = await response.json();
              // Map features to the format required by react-select { value: string, label: string }
              const formattedFeatures = data.features.map((feature: string) => ({ value: feature, label: feature }));
              setAvailableFeatures(formattedFeatures);
              setBasketPredictError(null); // Clear error on success
          } catch (err) {
              let errorMessage = 'An unknown error occurred.';
              if (err instanceof Error) {
                  errorMessage = err.message;
              }
              setBasketPredictError(`Failed to fetch prediction features: ${errorMessage}`);
              setAvailableFeatures([]); // Clear features on error
          } finally {
              setFeaturesLoading(false);
          }
      };
      fetchFeatures();
  }, [token]); // Depend on token

  // ML Prediction handler
  const handlePredictCLV = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredictLoading(true);
    setPredictError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict-clv`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          income_range: incomeRange,
          hh_size: parseInt(hhSize),
          children: parseInt(children),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }
      setPredictedCLV(data.predicted_clv);
    } catch (err) {
      if (err instanceof Error) {
        setPredictError(err.message || 'Prediction error');
      } else {
        setPredictError('Unknown prediction error.');
      }
      setPredictedCLV(null);
    } finally {
      setPredictLoading(false);
    }
  };
  
  // Basket Prediction Handler
  const handlePredictBasket = async (e: React.FormEvent) => {
      e.preventDefault();
      setBasketPredictLoading(true);
      setBasketPredictError(null);
      setBasketPredictResult(null);

      if (!token) {
          setBasketPredictError("Authentication token not found.");
          setBasketPredictLoading(false);
          return;
      }
      
      const commodityValues = selectedCommodities.map(option => option.value);
      if (commodityValues.length === 0) {
          setBasketPredictError("Please select at least one commodity currently in the basket.");
          setBasketPredictLoading(false);
          return;
      }
      
      try {
          const response = await fetch(`${API_BASE_URL}/predict-target-item`, {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
                  'Authorization': `Bearer ${token}` // Include auth token
              },
              body: JSON.stringify({ commodities: commodityValues }),
          });
          const data = await response.json();
          if (!response.ok) {
              throw new Error(data.detail || 'Basket prediction failed');
          }
          setBasketPredictResult(data);
      } catch (err) {
          if (err instanceof Error) {
              setBasketPredictError(err.message || 'Basket prediction error');
          } else {
              setBasketPredictError('Unknown basket prediction error.');
          }
          setBasketPredictResult(null);
      } finally {
          setBasketPredictLoading(false);
      }
  };

  // Handle selection change for the prediction input
  const handleSelectChange = (selectedOptions: MultiValue<{ value: string; label: string }>) => {
    setSelectedCommodities(selectedOptions ? selectedOptions.map((option: { value: string; label: string }) => ({ value: option.value, label: option.label })) : []);
  };

  // Custom styles for React Select (optional, for better dark mode etc.)
  const selectStyles: StylesConfig<{ value: string; label: string }, true> = {
    control: (provided: any) => ({
      ...provided,
      backgroundColor: '#333',
      borderColor: '#555',
    }),
    menu: (provided: any) => ({
      ...provided,
      backgroundColor: '#333',
    }),
    option: (provided: any, state: { isSelected: boolean; isFocused: boolean }) => ({
      ...provided,
      backgroundColor: state.isSelected ? '#555' : state.isFocused ? '#444' : '#333',
      color: 'white',
      ':active': {
        backgroundColor: '#666',
      },
    }),
    multiValue: (provided: any) => ({
      ...provided,
      backgroundColor: '#555',
    }),
    multiValueLabel: (provided: any) => ({
      ...provided,
      color: 'white',
    }),
    multiValueRemove: (provided: any) => ({
      ...provided,
      color: '#aaa',
      ':hover': {
        backgroundColor: '#c53030',
        color: 'white',
      },
    }),
    // Add other style overrides if needed (input, placeholder, etc.)
  };

  return (
    <div>
      <h2 className="text-3xl font-bold mb-8 text-indigo-800 border-b pb-3 border-indigo-200">Analytics Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <ChartWrapper title="Top 10 Spenders" fetchState={topSpenders}>
          <SimpleBarChart 
            data={topSpenders.data || []} 
            xAxisKey="Hshd_num" 
            barDataKey="total_spend" 
            fillColor="#8884d8"
          />
        </ChartWrapper>

        <ChartWrapper title="Loyalty Trends (Spend by Loyalty/Week)" fetchState={loyaltyTrends}>
          <LoyaltyTrendsChart data={loyaltyTrends.data || []} />
        </ChartWrapper>

        <ChartWrapper title="Avg. Spend by Income Bracket" fetchState={engagementByIncome}>
          <SimpleBarChart 
            data={engagementByIncome.data || []} 
            xAxisKey="income_bracket" 
            barDataKey="avg_spend" 
            fillColor="#82ca9d"
          />
        </ChartWrapper>

        <ChartWrapper title="Spend by Brand Type" fetchState={brandPreference}>
          <SimplePieChart 
            data={brandPreference.data || []} 
            nameKey="brand_type" 
            dataKey="total_spend"
          />
        </ChartWrapper>

        <ChartWrapper title="Top 10 Frequent Item Pairs (Commodity)" fetchState={frequentPairs}>
          <FrequentPairsTable data={frequentPairs.data || []} />
        </ChartWrapper>

        <ChartWrapper title="Top 10 Popular Products (Commodity)" fetchState={popularProducts}>
          <SimpleBarChart 
            data={popularProducts.data || []} 
            xAxisKey="commodity" 
            barDataKey="total_spend" 
            fillColor="#ffc658"
          />
        </ChartWrapper>

        <ChartWrapper title="Seasonal Sales Trends (Total Spend by Month)" fetchState={seasonalTrends}>
          <SimpleLineChart 
            data={formatSeasonalData(seasonalTrends.data)} 
            xAxisKey="monthLabel" 
            lineDataKey="total_spend" 
            strokeColor="#ff7300"
          />
        </ChartWrapper>

        {/* --- ML PREDICTION CARD --- */}
        <div className="mb-6 p-5 border rounded-lg bg-gradient-to-br from-gray-800 to-gray-900 shadow-md col-span-1 md:col-span-2">
          <h3 className="text-lg font-semibold mb-3 text-indigo-400">Predict Customer Lifetime Value (CLV)</h3>

          <form className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4" onSubmit={handlePredictCLV}>
            <input 
              type="text"
              placeholder="Income Range (e.g. 50-75k)"
              value={incomeRange}
              onChange={(e) => setIncomeRange(e.target.value)}
              className="p-2 rounded-md bg-gray-700 text-white focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
            <input 
              type="number"
              placeholder="Household Size (e.g. 4)"
              value={hhSize}
              onChange={(e) => setHhSize(e.target.value)}
              className="p-2 rounded-md bg-gray-700 text-white focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
            <input 
              type="number"
              placeholder="Children Count"
              value={children}
              onChange={(e) => setChildren(e.target.value)}
              className="p-2 rounded-md bg-gray-700 text-white focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
            <button
              type="submit"
              disabled={predictLoading}
              className="col-span-1 md:col-span-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 rounded-md shadow-md disabled:opacity-50 mt-2"
            >
              {predictLoading ? 'Predicting...' : 'Predict CLV'}
            </button>
          </form>

          {predictError && (
            <p className="text-red-500 text-sm">{predictError}</p>
          )}

          {predictedCLV !== null && (
            <p className="text-green-400 font-semibold text-lg">
              Predicted CLV: <span className="underline">${predictedCLV}</span>
            </p>
          )}
        </div>

        {/* --- NEW: Basket Item Prediction Card --- */}
        <div className="mb-6 p-5 border rounded-lg bg-gradient-to-br from-blue-50 to-indigo-100 shadow-md col-span-1 md:col-span-2">
            <h3 className="text-lg font-semibold mb-3 text-indigo-700">Predict Next Item (Cross-Sell Opportunity)</h3>
            <p className="text-sm text-gray-600 mb-4">Select commodities currently in the basket to predict the probability of adding DAIRY.</p>

            <form onSubmit={handlePredictBasket}>
                <label htmlFor="commodity-select" className="block text-sm font-medium text-gray-700 mb-1">Select Items in Basket:</label>
                <Select
                    id="commodity-select"
                    isMulti
                    options={availableFeatures}
                    value={selectedCommodities}
                    onChange={handleSelectChange}
                    isLoading={featuresLoading}
                    placeholder={featuresLoading ? "Loading commodities..." : "Select commodities..."}
                    className="mb-4"
                    styles={selectStyles}
                />
                
                <button
                    type="submit"
                    disabled={basketPredictLoading || featuresLoading || availableFeatures.length === 0}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 rounded-md shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {basketPredictLoading ? 'Predicting...' : 'Predict Probability of Adding DAIRY'}
                </button>
            </form>

            {basketPredictError && (
                <p className="text-red-500 text-sm mt-3">Error: {basketPredictError}</p>
            )}

            {basketPredictResult && (
                <div className="mt-4 p-3 bg-green-100 border border-green-300 rounded-md">
                    <p className="text-green-800 font-semibold">
                        Predicted Probability of adding {basketPredictResult.target_item}:
                        <span className="text-xl font-bold ml-2">{basketPredictResult.probability}%</span>
                    </p>
                     <p className="text-xs text-green-700 mt-1">Based on items: {selectedCommodities.map(o => o.label).join(', ') || '(None)'}</p>
                </div>
            )}
             {availableFeatures.length === 0 && !featuresLoading && !basketPredictError && (
                 <p className="text-gray-500 text-sm mt-3">Commodity list for prediction is unavailable. Model might not be loaded correctly.</p>
            )}
        </div>

      </div>

      {/* --- Churn Risk Section --- */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-6 text-red-700 border-b pb-3 border-red-200">Customer Churn Risk (Inactive > 8 Weeks)</h2>
        
        {/* --- Churn Summary Charts --- */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <ChartWrapper title="At-Risk Count by Loyalty Status" fetchState={churnRisk}>
            {churnRisk.data?.summary_stats?.count_by_loyalty ? (
               <SimpleBarChart 
                 data={churnRisk.data.summary_stats.count_by_loyalty}
                 xAxisKey="loyalty_flag" 
                 barDataKey="count" 
                 fillColor="#ef4444" // Red color for risk
               />
             ) : <p className="text-sm text-gray-500">Summary data unavailable.</p>}
          </ChartWrapper>
          
          <ChartWrapper title="At-Risk Count by Income Range" fetchState={churnRisk}>
             {churnRisk.data?.summary_stats?.count_by_income ? (
               <SimpleBarChart 
                 data={churnRisk.data.summary_stats.count_by_income}
                 xAxisKey="income_range" 
                 barDataKey="count" 
                 fillColor="#f87171" // Lighter red
               />
             ) : <p className="text-sm text-gray-500">Summary data unavailable.</p>}
          </ChartWrapper>
        </div>
        
        {/* --- At-Risk Customer Table --- */}
        <ChartWrapper title={`At-Risk Customer Details (${churnRisk.data?.at_risk_list?.length || 0})`} fetchState={churnRisk}>
          <div className="overflow-x-auto max-h-96"> 
             <table className="min-w-full divide-y divide-gray-200">
               <thead className="bg-gray-50">
                 <tr>
                   <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Household Num</th>
                   <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Purchase</th>
                   <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Loyalty</th>
                   <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Income Range</th>
                   <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">HH Size</th>
                   <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Children</th>
                 </tr>
               </thead>
               <tbody className="bg-white divide-y divide-gray-200">
                 {(churnRisk.data?.at_risk_list && churnRisk.data.at_risk_list.length > 0) ? (
                   churnRisk.data.at_risk_list.map((customer) => (
                     <tr key={customer.Hshd_num}>
                       <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{customer.Hshd_num}</td>
                       <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{customer.LastPurchaseDate}</td>
                       <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{customer.Loyalty_flag}</td>
                       <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{customer.IncomeRange}</td>
                       <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{customer.HshdSize}</td>
                       <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{customer.Children}</td>
                     </tr>
                   ))
                 ) : (
                   <tr>
                     <td colSpan={6} className="px-6 py-4 text-center text-sm text-gray-500">No customers identified at high risk of churn (based on 8-week inactivity).</td>
                   </tr>
                 )}
               </tbody>
             </table>
          </div>
        </ChartWrapper>
        
        {/* --- Retention Suggestions --- */}
        <div className="mt-6 p-5 border rounded-lg bg-yellow-50 shadow-sm">
           <h3 className="text-lg font-semibold mb-3 text-yellow-800">Potential Retention Strategies</h3>
           <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">
             <li>Segment offers based on loyalty status and income range (e.g., exclusive discounts for high-income/low-loyalty, reminders for loyal customers).</li>
             <li>Analyze purchase history of at-risk customers to personalize re-engagement campaigns (e.g., promotions on previously bought categories).</li>
             <li>Consider a targeted survey to understand reasons for disengagement.</li>
             <li>Review product assortment or pricing for commodities frequently purchased by the at-risk groups.</li>
           </ul>
        </div>
        
      </div>
    </div>
  );
};

export default DashboardPage;
