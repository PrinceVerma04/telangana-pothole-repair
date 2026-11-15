/** i18next setup. Telugu is the default UI language; Hindi and English selectable; choice persisted. */
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './en.json';
import hi from './hi.json';
import te from './te.json';
import { safeStorage } from '../lib/storage';

export const LANGS = [
  { code: 'te', label: 'తెలుగు' },
  { code: 'hi', label: 'हिन्दी' },
  { code: 'en', label: 'English' },
] as const;
export type LangCode = (typeof LANGS)[number]['code'];
export const LANG_KEY = 'gm.lang';
export const resources = { te: { translation: te }, hi: { translation: hi }, en: { translation: en } };

function initialLang(): LangCode {
  const s = safeStorage.get(LANG_KEY);
  return s === 'hi' || s === 'en' || s === 'te' ? s : 'te';
}

function applyDocLang(l: string) {
  if (typeof document !== 'undefined') {
    document.documentElement.lang = l;
    document.documentElement.dataset.lang = l;
  }
}

export function initI18n(lng: LangCode = initialLang()) {
  if (!i18n.isInitialized) {
    void i18n.use(initReactI18next).init({
      resources,
      lng,
      fallbackLng: 'en',
      interpolation: { escapeValue: false },
      returnNull: false,
    });
    applyDocLang(lng);
    i18n.on('languageChanged', applyDocLang);
  }
  return i18n;
}

export function setLang(l: LangCode) {
  safeStorage.set(LANG_KEY, l);
  void i18n.changeLanguage(l);
}

export default i18n;
