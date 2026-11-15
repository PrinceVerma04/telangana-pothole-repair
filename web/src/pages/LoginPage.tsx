import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router';
import { errorText, PageHeader } from '../components/ui';
import { homeFor } from '../lib/access';
import { getApi, getApiInfo } from '../lib/apiClient';
import { DEMO_USERS } from '../lib/demoUsers';
import { useSession } from '../lib/session';
import type { TokenRequest } from '../lib/types';

export default function LoginPage() {
  const { t } = useTranslation();
  const nav = useNavigate();
  const [sp] = useSearchParams();
  const setFromToken = useSession((s) => s.setFromToken);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const mock = getApiInfo()?.mode === 'mock';

  async function doLogin(req: TokenRequest, label: string) {
    setBusy(label);
    setErr(null);
    try {
      const tok = await getApi().login(req);
      setFromToken(tok, req.username ?? `+91${(req.phone ?? '').replace(/\D/g, '').slice(-10)}`);
      const next = sp.get('next');
      nav(next && next.startsWith('/') && !next.startsWith('/login') ? next : homeFor(tok.role), { replace: true });
    } catch (e) {
      setErr(errorText(e, t));
    } finally {
      setBusy(null);
    }
  }

  async function sendOtp() {
    setBusy('otp');
    setErr(null);
    try {
      await getApi().requestOtp(phone.trim());
      setOtpSent(true);
    } catch (e) {
      setErr(errorText(e, t));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="login">
      <PageHeader title={t('login.title')} sub={t('login.subtitle')} />
      {err ? <div className="alert alert-bad" role="alert">{err}</div> : null}
      <h2 className="h3">{t('login.pickRole')}</h2>
      <div className="role-grid">
        {DEMO_USERS.map((u) => (
          <button key={u.username} className="role-card" disabled={!!busy} data-testid={`login-${u.username}`}
            onClick={() => doLogin({ username: u.username, password: u.password }, u.username)}>
            <span className="role-name">{t(`role.${u.role}`)}</span>
            <span className="role-user">{u.username}</span>
            <span className="role-desc">{t(u.descKey)}</span>
            {busy === u.username ? <span className="muted small">{t('login.signingIn')}</span> : null}
          </button>
        ))}
      </div>
      <p className="muted small">{t('login.demoNote')}</p>
      <div className="grid-2">
        <form className="card card-b form" onSubmit={(e) => { e.preventDefault(); void doLogin({ username, password }, 'form'); }}>
          <h2 className="h3">{t('login.staff')}</h2>
          <label>{t('login.username')}<input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required /></label>
          <label>{t('login.password')}<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required /></label>
          <button className="btn btn-primary" disabled={!!busy}>{t('login.signIn')}</button>
        </form>
        <form className="card card-b form" onSubmit={(e) => { e.preventDefault(); if (otpSent) void doLogin({ phone: phone.trim(), otp: otp.trim() }, 'otp'); else void sendOtp(); }}>
          <h2 className="h3">{t('login.asCitizen')}</h2>
          <label>{t('login.phone')}<input inputMode="tel" placeholder="98xxxxxxxx" value={phone} onChange={(e) => setPhone(e.target.value)} required /></label>
          {otpSent ? <label>{t('login.otp')}<input inputMode="numeric" value={otp} onChange={(e) => setOtp(e.target.value)} required /></label> : null}
          {otpSent && mock ? <p className="muted small">{t('login.otpHint')}</p> : null}
          <button className="btn" disabled={!!busy}>{otpSent ? t('login.verifyOtp') : t('login.sendOtp')}</button>
        </form>
      </div>
    </div>
  );
}
