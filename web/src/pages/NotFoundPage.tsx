import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

export default function NotFoundPage() {
  const { t } = useTranslation();
  return (
    <div className="empty">
      <h1>404</h1>
      <p>{t('errors.notFound')}</p>
      <Link to="/" className="btn">{t('common.back')}</Link>
    </div>
  );
}
