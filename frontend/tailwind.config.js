/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBlue: '#0b1121',
        cardBlue: '#162036',
        accentBlue: '#4da3ff',
      },
    },
  },
  plugins: [],
}