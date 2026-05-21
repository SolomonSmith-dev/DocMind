# DocMind Web

React + Vite frontend for DocMind. Provides PDF upload, document management, and question answering with citations.

## Development

```bash
npm install
npm run dev
```

The dev server proxies `/api` and `/health` to `http://localhost:8000`.

## Build

```bash
npm run build
npm run preview
```

## E2E Tests

```bash
npm run test:e2e
```

Ensure the DocMind API is running locally before executing Playwright tests.
