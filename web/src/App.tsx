import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router';
import Layout, { Guard } from './components/Layout';
import { Loading } from './components/ui';
import LoginPage from './pages/LoginPage';
import NotFoundPage from './pages/NotFoundPage';

const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const MapPage = lazy(() => import('./pages/MapPage'));
const PotholePage = lazy(() => import('./pages/PotholePage'));
const QueuePage = lazy(() => import('./pages/QueuePage'));
const WorkOrdersPage = lazy(() => import('./pages/WorkOrdersPage'));
const RunPage = lazy(() => import('./pages/RunPage'));
const FieldPage = lazy(() => import('./pages/FieldPage'));
const ContractorPage = lazy(() => import('./pages/ContractorPage'));
const CitizenPage = lazy(() => import('./pages/CitizenPage'));
const AdminPage = lazy(() => import('./pages/AdminPage'));
const PublicWardsPage = lazy(() => import('./pages/PublicWardsPage'));

const S = ({ children }: { children: React.ReactNode }) => <Suspense fallback={<Loading />}>{children}</Suspense>;

export const routes = [
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Guard route="dashboard"><S><DashboardPage /></S></Guard> },
      { path: 'login', element: <LoginPage /> },
      { path: 'map', element: <Guard route="map"><S><MapPage /></S></Guard> },
      { path: 'potholes/:id', element: <Guard route="pothole"><S><PotholePage /></S></Guard> },
      { path: 'queue', element: <Guard route="queue"><S><QueuePage /></S></Guard> },
      { path: 'work-orders', element: <Guard route="workOrders"><S><WorkOrdersPage /></S></Guard> },
      { path: 'runs/:id', element: <Guard route="run"><S><RunPage /></S></Guard> },
      { path: 'field', element: <Guard route="field"><S><FieldPage /></S></Guard> },
      { path: 'contractors/:id', element: <Guard route="contractors"><S><ContractorPage /></S></Guard> },
      { path: 'citizen', element: <S><CitizenPage /></S> },
      { path: 'admin', element: <Guard route="admin"><S><AdminPage /></S></Guard> },
      { path: 'public/wards', element: <S><PublicWardsPage /></S> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

const router = createBrowserRouter(routes);

export default function App() {
  return <RouterProvider router={router} />;
}
