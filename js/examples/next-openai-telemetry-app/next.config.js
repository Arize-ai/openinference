/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
};

nextConfig.experimental = {
  instrumentationHook: true,
};

module.exports = nextConfig;
