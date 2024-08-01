/** @type {import('next').NextConfig} */

const isBuild = process.env.NEXTPROD === 'YES'

const nextConfig = {
  assetPrefix: isBuild ? './static' : undefined,
  reactStrictMode: true,
  output: 'export',
};

export default nextConfig;
