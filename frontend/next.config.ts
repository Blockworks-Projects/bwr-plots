import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ['pandas', 'numpy', 'plotly'],
  // Enable compression
  compress: true,
  // Optimize images
  images: {
    formats: ['image/webp', 'image/avif'],
  },
  // Proxy API requests to backend
  async rewrites() {
    return [
      {
        source: '/api/data/:path*',
        destination: 'http://localhost:8005/api/v1/data/:path*',
      },
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8005/api/v1/:path*',
      },
    ];
  },
  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ];
  },
};

export default nextConfig;
