/** @type {import('next').NextConfig} */

const isBuild = process.env.NEXTPROD === 'NO'

const nextConfig = {
  assetPrefix: isBuild ? undefined : './static',
  reactStrictMode: true,
  output: 'export',
};

export default nextConfig;
