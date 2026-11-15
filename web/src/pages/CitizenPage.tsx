import { useMutation } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, errorText, PageHeader } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtDateTime } from '../lib/format';
import { resizePhoto, type ResizedPhoto } from '../lib/image';
import type { CitizenReportOut } from '../lib/types';

export default function CitizenPage() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const [photo, setPhoto] = useState<ResizedPhoto | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [lat, setLat] = useState('');
  const [lon, setLon] = useState('');
  const [text, setText] = useState('');
  const [phone, setPhone] = useState('');
  const [locating, setLocating] = useState(false);
  const [locErr, setLocErr] = useState<string | null>(null);
  const [ticket, setTicket] = useState('');
  const [tracked, setTracked] = useState<CitizenReportOut | null>(null);

  useEffect(() => {
    if (!photo) { setPreview(null); return; }
    const u = URL.createObjectURL(photo.blob);
    setPreview(u);
    return () => URL.revokeObjectURL(u);
  }, [photo]);

  const submit = useMutation({
    mutationFn: () => getApi().createCitizenReport({
      lat: Number(lat), lon: Number(lon), text: text || undefined, lang: (['te', 'hi', 'en'].includes(L) ? L : 'te') as 'te' | 'hi' | 'en',
      phone: phone || undefined, photo: photo?.blob ?? null, photo_name: 'report.jpg',
    }),
  });
  const track = useMutation({ mutationFn: (tk: string) => getApi().getCitizenReport(tk), onSuccess: setTracked });

  function locate() {
    setLocErr(null);
    if (!('geolocation' in navigator)) { setLocErr(t('citizen.gpsUnavailable')); return; }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (p) => { setLat(p.coords.latitude.toFixed(6)); setLon(p.coords.longitude.toFixed(6)); setLocating(false); },
      () => { setLocErr(t('citizen.gpsUnavailable')); setLocating(false); },
      { enableHighAccuracy: true, timeout: 10_000 },
    );
  }

  return (
    <div className="citizen">
      <PageHeader title={t('citizen.title')} sub={t('citizen.subtitle')} />
      <div className="grid-2">
        <Card title={t('citizen.report')}>
          {submit.data ? (
            <div className="alert alert-good" role="status" data-testid="ticket">
              <p>{t('citizen.thanks', { ticket: submit.data.ticket })}</p>
              <p className="small">{submit.data.pothole_id ? t('citizen.linked', { id: submit.data.pothole_id }) : t('citizen.received')}</p>
              <button className="btn btn-sm" onClick={() => { setTicket(submit.data!.ticket); track.mutate(submit.data!.ticket); }}>{t('citizen.trackBtn')}</button>
              <button className="btn btn-sm" onClick={() => { submit.reset(); setPhoto(null); setText(''); }}>{t('citizen.another')}</button>
            </div>
          ) : (
            <form className="form" onSubmit={(e) => { e.preventDefault(); submit.mutate(); }}>
              <label className="btn btn-block">
                {photo ? t('field.retake') : t('citizen.photo')}
                <input type="file" accept="image/*" capture="environment" hidden onChange={async (e) => {
                  const f = e.target.files?.[0];
                  e.target.value = '';
                  if (f) setPhoto(await resizePhoto(f));
                }} />
              </label>
              {preview ? <img className="preview" src={preview} alt={t('citizen.photo')} /> : null}
              <fieldset>
                <legend>{t('citizen.location')}</legend>
                <button type="button" className="btn" onClick={locate} disabled={locating}>{locating ? t('citizen.locating') : t('citizen.useGps')}</button>
                {locErr ? <div className="muted small">{locErr}</div> : null}
                <div className="grid-2 tight">
                  <label>{t('citizen.lat')}<input inputMode="decimal" value={lat} onChange={(e) => setLat(e.target.value)} required placeholder="17.4375" /></label>
                  <label>{t('citizen.lon')}<input inputMode="decimal" value={lon} onChange={(e) => setLon(e.target.value)} required placeholder="78.4482" /></label>
                </div>
              </fieldset>
              <label>{t('citizen.description')}<textarea rows={3} maxLength={2000} value={text} onChange={(e) => setText(e.target.value)} placeholder={t('citizen.descPh')} /></label>
              <label>{t('citizen.phone')}<input inputMode="tel" value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="98xxxxxxxx" /></label>
              <p className="muted small">{t('citizen.privacy')}</p>
              {submit.error ? <div className="alert alert-bad">{errorText(submit.error, t)}</div> : null}
              <button className="btn btn-primary btn-block" disabled={submit.isPending}>{submit.isPending ? t('citizen.submitting') : t('citizen.submit')}</button>
            </form>
          )}
        </Card>
        <Card title={t('citizen.track')}>
          <form className="form" onSubmit={(e) => { e.preventDefault(); track.mutate(ticket.trim()); }}>
            <label>{t('citizen.ticket')}<input value={ticket} onChange={(e) => setTicket(e.target.value)} placeholder="GM-000001" required /></label>
            <button className="btn">{t('citizen.trackBtn')}</button>
          </form>
          {track.error ? <div className="alert alert-bad">{errorText(track.error, t)}</div> : null}
          {tracked ? (
            <dl className="kv" data-testid="tracked">
              <dt>{t('citizen.ticket')}</dt><dd>{tracked.ticket}</dd>
              <dt>{t('common.status')}</dt><dd><strong>{t(`citizen.status.${tracked.status}`, { defaultValue: tracked.status })}</strong></dd>
              <dt>{t('citizen.reportedAt')}</dt><dd>{fmtDateTime(tracked.created_at, L)}</dd>
              <dt>{t('citizen.location')}</dt><dd className="num">{tracked.lat.toFixed(5)}, {tracked.lon.toFixed(5)}</dd>
              {tracked.pothole_id ? <><dt>{t('citizen.potholeRef')}</dt><dd>#{tracked.pothole_id}</dd></> : null}
            </dl>
          ) : null}
        </Card>
      </div>
    </div>
  );
}
