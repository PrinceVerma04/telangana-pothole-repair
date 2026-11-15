import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, Navigate, NavLink, Outlet, useLocation, useNavigate } from 'react-router';
import { useRegisterSW } from 'virtual:pwa-register/react';
import { LANGS, setLang, type LangCode } from '../i18n';
import { canSee, type RouteKey } from '../lib/access';
import { getApiInfo, switchMode } from '../lib/apiClient';
import { useOnline } from '../lib/hooks';
import { useSession } from '../lib/session';

interface NavItem { to: string; key: RouteKey; label: string }

export function LangSwitch() {
  const { i18n } = useTranslation();
  return (
    <div className="lang" role="group" aria-label="Language / భాష / भाषा">
      {LANGS.map((l) => (
        <button key={l.code} type="button" lang={l.code} className={i18n.language === l.code ? 'on' : ''} aria-pressed={i18n.language === l.code}
          onClick={() => setLang(l.code as LangCode)}>{l.label}</button>
      ))}
    </div>
  );
}

function ModeBadge() {
  const { t } = useTranslation();
  const info = getApiInfo();
  if (!info) return null;
  const mock = info.mode === 'mock';
  return (
    <button type="button" className={`mode ${mock ? 'mode-mock' : 'mode-live'}`} title={`${t('common.modeReason')}: ${info.reason}`}
      onClick={() => { if (window.confirm(mock ? t('common.switchToLive') : t('common.switchToMock'))) switchMode(mock ? 'http' : 'mock'); }}>
      {mock ? t('common.mockMode') : t('common.liveMode')}
    </button>
  );
}

function PwaPrompt() {
  const { t } = useTranslation();
  const { needRefresh: [needRefresh, setNeedRefresh], updateServiceWorker } = useRegisterSW({ immediate: true });
  if (!needRefresh) return null;
  return (
    <div className="toast" role="status">
      {t('pwa.update')}
      <button className="btn btn-sm btn-primary" onClick={() => updateServiceWorker(true)}>{t('pwa.reload')}</button>
      <button className="btn btn-sm" onClick={() => setNeedRefresh(false)}>{t('common.close')}</button>
    </div>
  );
}

export default function Layout() {
  const { t } = useTranslation();
  const s = useSession();
  const online = useOnline();
  const nav = useNavigate();
  const loc = useLocation();
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [loc.pathname]);

  const items: NavItem[] = [
    { to: '/', key: 'dashboard', label: t('nav.dashboard') },
    { to: '/map', key: 'map', label: t('nav.map') },
    { to: '/queue', key: 'queue', label: t('nav.queue') },
    { to: '/work-orders', key: 'workOrders', label: t('nav.workOrders') },
    { to: '/field', key: 'field', label: t('nav.field') },
    { to: `/contractors/${s.contractor_id ?? 1}`, key: 'contractors', label: t('nav.contractors') },
    { to: '/admin', key: 'admin', label: t('nav.admin') },
    { to: '/citizen', key: 'citizen', label: t('nav.citizen') },
    { to: '/public/wards', key: 'publicWards', label: t('nav.publicWards') },
  ].filter((i) => canSee(i.key as RouteKey, s.role)) as NavItem[];

  return (
    <div className="app">
      <header className="topbar no-print">
        <div className="topbar-row">
          <Link to="/" className="brand" aria-label="GunthaMeter">
            <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden><rect width="32" height="32" rx="6" fill="#0b3d6e" /><ellipse cx="16" cy="19" rx="10" ry="5" fill="#f2c94c" /><ellipse cx="16" cy="18" rx="6" ry="2.6" fill="#0b3d6e" /><rect x="15" y="5" width="2" height="9" fill="#fff" /></svg>
            <span className="brand-name">{t('app.name')}</span>
            <span className="brand-sub">{t('app.org')}</span>
          </Link>
          <div className="topbar-tools">
            {!online ? <span className="chip tone-bad">{t('common.offline')}</span> : null}
            <ModeBadge />
            <LangSwitch />
            {s.token ? (
              <div className="user">
                <span className="user-name">{s.username}<small>{t(`role.${s.role}`)}</small></span>
                <button className="btn btn-sm" onClick={() => { s.logout(); nav('/login'); }}>{t('nav.logout')}</button>
              </div>
            ) : (
              <Link className="btn btn-sm btn-primary" to={`/login?next=${encodeURIComponent(loc.pathname)}`}>{t('nav.login')}</Link>
            )}
            <button className="btn btn-sm menu-btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>{t('nav.menu')}</button>
          </div>
        </div>
        <nav className={`nav ${open ? 'open' : ''}`} aria-label={t('nav.menu')}>
          {items.map((i) => (
            <NavLink key={i.key} to={i.to} end={i.to === '/'}>{i.label}</NavLink>
          ))}
        </nav>
      </header>
      <main className="main">
        <Outlet />
      </main>
      <footer className="footer no-print">
        <span>{t('app.footer')}</span>
        <span>Map © <a href="https://openfreemap.org" target="_blank" rel="noreferrer">OpenFreeMap</a> © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a></span>
      </footer>
      <PwaPrompt />
    </div>
  );
}

/** Route guard: redirects to /login when signed out; shows "forbidden" for the wrong role. */
export function Guard({ route, children }: { route: RouteKey; children: React.ReactNode }) {
  const { t } = useTranslation();
  const s = useSession();
  const loc = useLocation();
  if (!s.token) return <Navigate to={`/login?next=${encodeURIComponent(loc.pathname + loc.search)}`} replace />;
  if (!canSee(route, s.role)) return <div className="alert alert-bad">{t('errors.forbidden')}</div>;
  return <>{children}</>;
}
