import '@fontsource/noto-sans/latin-400.css';
import '@fontsource/noto-sans/latin-600.css';
import '@fontsource/noto-sans-telugu/telugu-400.css';
import '@fontsource/noto-sans-telugu/telugu-600.css';
import '@fontsource/noto-sans-devanagari/devanagari-400.css';
import '@fontsource/noto-sans-devanagari/devanagari-600.css';
import './styles.css';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { initI18n } from './i18n';
import { isRetryable } from './lib/api';
import { bootstrapApi } from './lib/apiClient';
import { fieldDb } from './lib/offline/db';
import { OfflineQueue } from './lib/offline/queue';
import { startAutoSync } from './lib/offline/sync';
import { getApi } from './lib/apiClient';
import { setQueue } from './lib/offline/instance';
import { registerServiceWorker } from './lib/pwa';

initI18n();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, refetchOnWindowFocus: false, retry: (n, e) => n < 2 && isRetryable(e) },
  },
});

async function start() {
  const root = createRoot(document.getElementById('root')!);
  root.render(<div className="boot">GunthaMeter · గుంత మీటర్ …</div>);
  const info = await bootstrapApi();
  document.documentElement.dataset.apiMode = info.mode;
  const queue = new OfflineQueue(fieldDb());
  setQueue(queue);
  startAutoSync(queue, () => getApi());
  void registerServiceWorker();
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </StrictMode>,
  );
}

void start();
