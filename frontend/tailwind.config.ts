import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      keyframes: {
        "color-fade": {
          "0%": { "color": "red" },
          "100%": { "color": "black" },
        }
      },
      animation: {
        "color-fade": "color-fade 0.75s ease-out",
      }
    }
  },
  plugins: [
    require('daisyui'),
  ],
  daisyui: {
    themes: ["bumblebee"],
  },
};
export default config;
