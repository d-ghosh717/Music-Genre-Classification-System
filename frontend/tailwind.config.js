/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        academic: {
          bg: "#F7F7F5",
          text: "#202020",
          secondary: "#737373",
          border: "#E5E5E5",
          card: "#FFFFFF",
          accent: "#2563EB",
          accentMuted: "#3B82F6",
          darkMuted: "#374151"
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
