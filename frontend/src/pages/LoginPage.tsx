import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';

const LoginPage: React.FC = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState(''); // Only for registration
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/dashboard"; // Redirect back or to search page

  const API_BASE_URL = 'https://dev-cc.onrender.com'; // Move to config/env later

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }
      
      // Assuming backend returns { message: "...", username: "..." }
      // We'll simulate a token for now, as backend doesn't return one
      const fakeToken = "fake-jwt-token"; 
      login(data.username, fakeToken); // Update context
      navigate(from, { replace: true }); // Redirect after login

    } catch (err) {
      if (err instanceof Error) {
          setError(err.message || 'An error occurred during login.');
      } else {
          setError('An unknown error occurred during login.')
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
        const response = await fetch(`${API_BASE_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password }),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Registration failed');
        }

        // Optionally log the user in automatically after registration
        // Or just show a success message and let them log in
        alert('Registration successful! Please log in.'); 
        setIsRegistering(false); // Switch back to login view
        // Clear fields?
        setUsername('');
        setPassword('');
        setEmail('');

    } catch (err) {
        if (err instanceof Error) {
            setError(err.message || 'An error occurred during registration.');
        } else {
             setError('An unknown error occurred during registration.')
        }
    } finally {
        setLoading(false);
    }
};


  return (
    <div className="flex items-center justify-center ">
      <div className="w-full max-w-md p-8 space-y-6 bg-gray-50 rounded-xl shadow-lg border border-gray-200">
        <h2 className="text-3xl font-bold text-center text-indigo-700">
          {isRegistering ? 'Create Account' : 'Welcome Back'}
        </h2>
        <form className="space-y-6" onSubmit={isRegistering ? handleRegister : handleLogin}>
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-700">
              Username
            </label>
            <input
              id="username"
              name="username"
              type="text"
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>

          {isRegistering && (
             <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          )}

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

           {error && (
              <p className="text-sm text-red-600">{error}</p>
           )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed transition-colors duration-150 ease-in-out"
            >
              {loading ? 'Processing...' : (isRegistering ? 'Register' : 'Login')}
            </button>
          </div>
        </form>

        <div className="text-sm text-center">
          <button
            type="button"
            onClick={() => {
                setIsRegistering(!isRegistering);
                setError(null); // Clear errors on switch
            }}
            className="font-medium text-indigo-600 hover:text-indigo-500"
          >
            {isRegistering ? 'Already have an account? Login' : "Don't have an account? Register"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage; 