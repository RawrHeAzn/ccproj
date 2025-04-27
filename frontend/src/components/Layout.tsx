import React from 'react';
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext'; // Import useAuth

const navLinks = [
  { href: '/dashboard', text: 'Dashboard' },
  { href: '/search', text: 'Search' },
  { href: '/upload-data', text: 'Upload Data' }, // Add link for data upload
];

const Layout: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout, isLoading } = useAuth(); // Get user, logout, isLoading

  const handleLogout = () => {
    logout();
    navigate('/login'); // Redirect to login after logout
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <nav className="bg-gradient-to-r from-indigo-600 to-purple-700 shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <span className="flex-shrink-0 flex items-center text-2xl font-bold text-white mr-8">
                Retail Portal
              </span>
              <div className="flex space-x-6">
                {navLinks.map((link) => (
                  <Link
                    key={link.href}
                    to={link.href}
                    className={`inline-flex items-center px-3 py-1.5 border border-transparent text-sm font-medium rounded-md transition-colors duration-150 ease-in-out 
                       ${location.pathname.startsWith(link.href)
                        ? 'bg-white text-indigo-700' // Active state
                        : 'text-indigo-100 bg-indigo-600 hover:bg-indigo-500 hover:text-white' // Inactive state
                       }`}
                    aria-current={location.pathname.startsWith(link.href) ? 'page' : undefined}
                  >
                    {link.text}
                  </Link>
                ))}
              </div>
            </div>
            <div className="flex items-center">
              {isLoading ? (
                 <span className="text-sm text-indigo-100">Loading user...</span>
              ) : user ? (
                <>
                  <span className="text-sm font-medium text-indigo-100 mr-4">Welcome, {user}!</span>
                  <button
                    onClick={handleLogout}
                    className="inline-flex items-center px-2 py-1 border border-transparent text-xs font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-indigo-700 focus:ring-white transition-colors duration-150 ease-in-out"
                  >
                    Logout
                  </button>

                </>
              ) : (
                 <Link to="/login" className="inline-flex items-center px-3 py-1.5 border border-transparent text-sm font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50">
                   Login
                 </Link>
              )}
            </div>
          </div>
        </div>
      </nav>

      <main className="py-10">
        <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
          <div className="bg-white/90 backdrop-blur-sm shadow-xl rounded-lg overflow-hidden border border-gray-200">
            <div className="px-4 py-8 sm:p-10 min-h-[calc(100vh-12rem)]">
              <Outlet /> 
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Layout; 