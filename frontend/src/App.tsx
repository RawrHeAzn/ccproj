// import { useState } from 'react' // Removed unused import
// import reactLogo from './assets/react.svg' // Removed unused import
// import viteLogo from '/vite.svg' // Removed unused import
// import './App.css' // Removed default CSS import

import React from 'react'; // Ensure React is imported for JSX types
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation
} from "react-router-dom";
import { useAuth } from './context/AuthContext'; // Import useAuth hook

import Layout from './components/Layout';
import LoginPage from './pages/LoginPage';
import SearchPage from './pages/SearchPage';
// import UploadPage from './pages/UploadPage'; // Remove this unused import
import DashboardPage from './pages/DashboardPage';
import DataUploadPage from './pages/DataUploadPage'; // Import the new page

// Protected route component
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { user, isLoading, token } = useAuth(); // Use auth context
  const location = useLocation(); // Import useLocation from react-router-dom if not already

  if (isLoading) {
     // Show a loading indicator while checking auth status
     return <div className="flex justify-center items-center h-screen"><p>Loading...</p></div>;
  }

  if (!user || !token) { // Check for user/token from context
    // Redirect them to the /login page, saving the current location
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return children;
};

function App() {
  // const [count, setCount] = useState(0) // Removed default state

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        
        {/* Routes protected by Layout */}
        <Route element={<Layout />}>
          <Route 
            path="/search" 
            element={
              <ProtectedRoute>
                 <SearchPage />
               </ProtectedRoute>
            }
          />
          {/* Removed Upload Route */}
          {/* <Route 
             path="/upload" 
             element={
               <ProtectedRoute>
                 <UploadPage />
               </ProtectedRoute>
             }
           /> */}
          <Route 
             path="/dashboard" 
             element={
               <ProtectedRoute>
                 <DashboardPage />
               </ProtectedRoute>
             }
           />
          {/* --- NEW ROUTE FOR DATA UPLOAD --- */}
          <Route 
             path="/upload-data" 
             element={
               <ProtectedRoute>
                 <DataUploadPage />
               </ProtectedRoute>
             }
           />
           {/* Optional: Redirect root path to login or search */}
           <Route path="/" element={<Navigate to="/login" replace />} /> 
        </Route>

        {/* Handle 404 or redirect */}
        {/* <Route path="*" element={<NotFoundPage />} /> */}
        <Route path="*" element={<Navigate to="/login" replace />} /> 

      </Routes>
    </Router>
  );
}

export default App
