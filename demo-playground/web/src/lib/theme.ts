'use client';

import { useEffect, useState } from 'react';

/**
 * Whether the page is currently dark: `<html data-theme="dark|light">` wins,
 * otherwise the system preference (the same rule globals.css uses).
 */
export function useIsDark(): boolean {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const compute = () => {
      const forced = document.documentElement.getAttribute('data-theme');
      setDark(forced ? forced === 'dark' : mq.matches);
    };
    compute();
    mq.addEventListener('change', compute);
    const obs = new MutationObserver(compute);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => {
      mq.removeEventListener('change', compute);
      obs.disconnect();
    };
  }, []);
  return dark;
}
