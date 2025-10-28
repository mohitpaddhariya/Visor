import { defineConfig } from '@hey-api/openapi-ts';

const apiBaseUrl = 'http://localhost:8003';

const input = apiBaseUrl + '/openapi.json';

export default defineConfig({
  input,
  output: {
    format: 'prettier',
    lint: 'eslint',
    path: './api-client',
  },
  plugins: [
    '@hey-api/client-axios',
    '@hey-api/schemas',
    {
      dates: true,
      name: '@hey-api/transformers',
    },
    {
      enums: 'javascript',
      name: '@hey-api/typescript',
    },
    {
      name: '@hey-api/sdk',
      transformer: true,
    },
  ],
});
