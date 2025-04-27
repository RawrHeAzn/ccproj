import React, { createContext, useState, useContext, ReactNode, useEffect } from 'react';

// Define the shape of the context data
interface AuthContextType {
  user: string | null;
  token: string | null;
  isLoading: boolean;
  login: (username: string, token: string) => void;
  logout: () => void;
}

// Create the context with a default value (can be undefined or a default object)
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Create the provider component
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true); // Start loading until checked

  // Check local storage on initial load
  useEffect(() => {
    try {
      const storedToken = localStorage.getItem('authToken');
      const storedUser = localStorage.getItem('authUser');
      if (storedToken && storedUser) {
        setToken(storedToken);
        setUser(storedUser);
      }
    } catch (error) {
      console.error("Failed to access localStorage:", error);
      // Handle cases where localStorage might be disabled or unavailable
    } finally {
        setIsLoading(false); // Finished checking
    }
  }, []);

  const login = (username: string, jwtToken: string) => {
     try {
        localStorage.setItem('authUser', username);
        localStorage.setItem('authToken', jwtToken); // In a real app, the backend would return a token
        setUser(username);
        setToken(jwtToken);
     } catch (error) {
       console.error("Failed to save auth state to localStorage:", error);
     }
  };

  const logout = () => {
    try {
        localStorage.removeItem('authUser');
        localStorage.removeItem('authToken');
        setUser(null);
        setToken(null);
    } catch (error) {
      console.error("Failed to remove auth state from localStorage:", error);
    }
  };

  // Value provided to consuming components
  const value = { user, token, isLoading, login, logout };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use the auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 