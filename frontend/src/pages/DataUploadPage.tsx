import React, { useState, ChangeEvent } from 'react';
import { useAuth } from '../context/AuthContext'; // To get the token for authentication

// Type for individual file upload state
interface FileUploadState {
  file: File | null;
  status: 'idle' | 'uploading' | 'success' | 'error' | 'polling'; // Added polling status
  message: string | null;
}

// Type for overall process state
type UploadStep = 'households' | 'products' | 'transactions' | 'done';

const DataUploadPage: React.FC = () => {
  const { token } = useAuth();
  const API_BASE_URL = 'https://dev-cc-omega.vercel.app';

  // State for each file (still needed to hold the file object and status)
  const [fileStates, setFileStates] = useState<Record<UploadStep, FileUploadState>>({
    households: { file: null, status: 'idle', message: null },
    products: { file: null, status: 'idle', message: null },
    transactions: { file: null, status: 'idle', message: null },
    done: { file: null, status: 'idle', message: null } // Placeholder
  });
  
  // State for current step in the process
  const [currentStep, setCurrentStep] = useState<UploadStep>('households');
  
  // State for dashboard update indicator (global for the page)
  const [isDashboardUpdating, setIsDashboardUpdating] = useState(false);
  let pollingIntervalId: NodeJS.Timeout | null = null; // Keep track of polling interval

  // File selection handler - updates the specific file state
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files ? event.target.files[0] : null;
    setFileStates(prev => ({
      ...prev,
      [currentStep]: { file: file, status: 'idle', message: file ? 'File selected' : null } 
    }));
  };

  // Upload function for the *current step's* file
  const uploadCurrentFile = async () => {
    const stepState = fileStates[currentStep];
    const endpointMap: Record<string, string> = {
        households: '/upload/households',
        products: '/upload/products',
        transactions: '/upload/transactions'
    };
    const endpoint = endpointMap[currentStep];

    if (!stepState || !endpoint) {
        console.error("Invalid step or endpoint");
        setFileStates(prev => ({ ...prev, [currentStep]: {...stepState, status:'error', message:'Internal error: Invalid step.'}}));
        return;
    }

    if (!stepState.file) {
      setFileStates(prev => ({ ...prev, [currentStep]: {...stepState, status: 'error', message: 'No file selected.'}}));
      return;
    }
    if (!token) {
      setFileStates(prev => ({ ...prev, [currentStep]: {...stepState, status: 'error', message: 'Authentication token not found.'}}));
      return;
    }

    // Update status to uploading
    setFileStates(prev => ({ ...prev, [currentStep]: {...stepState, status: 'uploading', message: 'Uploading...'}}));

    const formData = new FormData();
    formData.append('file', stepState.file);

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Upload failed with status: ${response.status}`);
      }
      
      // Backend confirmed successful processing & triggered update
      // Update status, show backend message, start polling, and move to next step
      setFileStates(prev => ({ ...prev, [currentStep]: {...stepState, status: 'polling', message: data.message || 'Upload processed, updating dashboard...' }}));
      startPollingDashboardStatus(); // Start global polling indicator
      advanceStep(); // Move to the next step
      
    } catch (err) {
      let errorMessage = 'An unknown error occurred.';
      if (err instanceof Error) {
        errorMessage = err.message;
      }
       // Update status to error
      setFileStates(prev => ({ ...prev, [currentStep]: {...stepState, status: 'error', message: `Upload failed: ${errorMessage}`}}));
    }
  };

  // Function to advance to the next step
  const advanceStep = () => {
      if (currentStep === 'households') setCurrentStep('products');
      else if (currentStep === 'products') setCurrentStep('transactions');
      else if (currentStep === 'transactions') setCurrentStep('done');
  };

  // Polling function - simplified to just manage the global indicator
  const startPollingDashboardStatus = () => {
      if (pollingIntervalId) clearInterval(pollingIntervalId); // Clear previous interval if any
      setIsDashboardUpdating(true); // Show indicator
      
      pollingIntervalId = setInterval(async () => {
          try {
              const response = await fetch(`${API_BASE_URL}/dashboard-update-status`);
              if (!response.ok) {
                  console.error("Polling error: Status check failed");
                  // Stop polling on error maybe?
                  // clearInterval(pollingIntervalId!);
                  // setIsDashboardUpdating(false);
                  return; 
              }
              const data = await response.json();
              if (data.updating === false) {
                  clearInterval(pollingIntervalId!);
                  pollingIntervalId = null;
                  setIsDashboardUpdating(false); // Hide indicator
                  console.log("Dashboard update complete (polled).");
                  // We don't need to update individual file statuses here anymore
              }
          } catch (error) {
              console.error("Polling error:", error);
              clearInterval(pollingIntervalId!); // Stop polling on error
              pollingIntervalId = null;
              setIsDashboardUpdating(false); 
              // Maybe show a general polling error message?
          }
      }, 3000); // Poll every 3 seconds
  };

  // Function to reset the entire process
   const handleReset = () => {
       if (pollingIntervalId) clearInterval(pollingIntervalId);
       setIsDashboardUpdating(false);
       setCurrentStep('households');
       setFileStates({
           households: { file: null, status: 'idle', message: null },
           products: { file: null, status: 'idle', message: null },
           transactions: { file: null, status: 'idle', message: null },
           done: { file: null, status: 'idle', message: null } 
       });
   };

  // Helper to render the input for the current step
  const renderCurrentStepInput = () => {
    const stepConfig = {
        households: { label: 'Step 1: Upload Households Data (.csv)', required: true },
        products: { label: 'Step 2: Upload Products Data (.csv)', required: true },
        transactions: { label: 'Step 3: Upload Transactions Data (.csv)', required: true },
        done: { label: 'Finished!', required: false}
    };

    if (currentStep === 'done') {
      return (
        <div className="p-4 text-center bg-green-50 border border-green-200 rounded-lg">
          <p className="font-semibold text-green-700">All steps processed.</p>
          <p className="text-sm text-gray-600">Check the status messages above for details. Dashboard data may still be updating in the background.</p>
        </div>
      );
    }
    
    const config = stepConfig[currentStep];
    const state = fileStates[currentStep];
    const isUploadingCurrent = state.status === 'uploading' || state.status === 'polling';

    let statusColor = 'text-gray-600';
    if (state.status === 'success' || state.status === 'polling') statusColor = 'text-green-600'; // Treat polling as temp success
    if (state.status === 'error') statusColor = 'text-red-600';
    if (state.status === 'uploading') statusColor = 'text-blue-600';
    
    return (
      <div className="mb-6 p-4 border border-indigo-200 rounded-lg bg-white shadow-sm">
        <label className="block text-lg font-semibold text-indigo-700 mb-3">
          {config.label}
        </label>
        <input 
          type="file" 
          accept=".csv" 
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 disabled:opacity-50 mb-3"
          disabled={isUploadingCurrent || isDashboardUpdating}
          key={currentStep} // Force re-render on step change to clear file input visually
        />
        {state.message && (
          <p className={`text-xs mb-3 ${statusColor}`}>
            {state.status === 'uploading' && <span className="animate-pulse">(Uploading file...) </span>}
             {state.status === 'polling' && <span className="animate-pulse">(Processing & updating dashboard...) </span>}
            {state.message}
          </p>
        )}
        <div className="flex space-x-3">
             <button 
                 type="button" // Prevent form submission
                 onClick={uploadCurrentFile} 
                 className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                 disabled={!state.file || isUploadingCurrent || isDashboardUpdating}
             >
                 {isUploadingCurrent ? 'Processing...' : `Upload ${currentStep.charAt(0).toUpperCase() + currentStep.slice(1)}`}
             </button>
            {/* Add Skip Button for all steps except 'done' */}
            <button 
                type="button" 
                onClick={advanceStep} // Simply advances to the next step
                className="flex-1 py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                disabled={isUploadingCurrent || isDashboardUpdating} // Disable during upload/polling
            >
                Skip
            </button>
         </div>
      </div>
    );
  };

  return (
    <div className="max-w-2xl mx-auto mt-8 p-6 relative">
      
      {/* Dashboard Updating Indicator Overlay */} 
      {isDashboardUpdating && (
          <div className="absolute inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50 rounded-lg">
              <div className="bg-white p-6 rounded-lg shadow-xl text-center">
                  <p className="text-lg font-semibold text-indigo-700 animate-pulse">Updating Dashboard Data...</p>
                  <p className="text-sm text-gray-600 mt-2">Please wait, this might take a minute.</p>
              </div>
          </div>
      )}

      <h2 className="text-3xl font-bold mb-6 text-indigo-800 border-b pb-2 border-indigo-200">Upload Datasets</h2>
      <p className="text-sm text-gray-600 mb-6">
        Upload datasets sequentially. Uploading will 
        <strong className="text-orange-600">append</strong> the new data to the existing database tables.
      </p>

      {/* Render current step */} 
      {renderCurrentStepInput()}

       {/* Only show reset button if process has started */} 
       {(currentStep !== 'households' || fileStates.households.file) && (
         <button 
             type="button" 
             onClick={handleReset}
             className="mt-6 w-full py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
             disabled={isDashboardUpdating || fileStates[currentStep]?.status === 'uploading' || fileStates[currentStep]?.status === 'polling'}
         >
             Reset Upload Process
         </button>
        )}

    </div>
  );
};

export default DataUploadPage; 