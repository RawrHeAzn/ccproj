/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}", // Include React component files
    ],
    theme: {
        extend: {
          colors: {
            'gray-900': '#111827', // Dark gray, for example
          },
          backgroundImage: {
            'gradient-to-br': 'linear-gradient(to bottom right, var(--tw-gradient-stops))',
            'gradient-to-r': 'linear-gradient(to right, var(--tw-gradient-stops))',
          }
        },
      },
    plugins: [],
}