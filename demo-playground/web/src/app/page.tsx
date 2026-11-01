'use client';

import { useCallback, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import SiteHeader from '@/components/ui/SiteHeader';
import PlaygroundBar, { type PlaygroundTab, type ServerStatus } from '@/components/ui/PlaygroundBar';
import HostedBanner from '@/components/ui/HostedBanner';
import { readZenFromUrl } from '@/lib/recording';

// Client-only: the notebook uses a Web Worker + CodeMirror, the Math tab localStorage.
const Notebook = dynamic(() => import('@/components/notebook/Notebook'), { ssr: false });
const MathNotebook = dynamic(() => import('@/components/math/MathNotebook'), { ssr: false });

/** `?tab=python|math`; a `?demo=` recording payload always targets the Python notebook. */
function initialTab(): PlaygroundTab {
  if (typeof window === 'undefined') return 'math';
  const q = new URLSearchParams(window.location.search);
  const t = q.get('tab');
  if (t === 'python' || t === 'math') return t;
  if (q.has('demo')) return 'python';
  return 'math';
}

export default function PlaygroundPage() {
  // null until the URL has been read (keeps the prerendered HTML and the first
  // client render identical, and avoids mounting a tab the URL did not ask for).
  const [tab, setTab] = useState<PlaygroundTab | null>(null);
  const [mounted, setMounted] = useState<Record<PlaygroundTab, boolean>>({ math: false, python: false });
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [pythonStatus, setPythonStatus] = useState<ServerStatus>('unknown');
  const [mathStatus, setMathStatus] = useState<ServerStatus>('unknown');
  const [mathLabel, setMathLabel] = useState<string | undefined>(undefined);
  const onMathStatus = useCallback((status: ServerStatus, label?: string) => {
    setMathStatus(status);
    setMathLabel(label);
  }, []);
  const [isDirty, setIsDirty] = useState(false);
  const [zenMode] = useState(readZenFromUrl);

  useEffect(() => {
    const t = initialTab();
    setTab(t);
    setMounted((m) => ({ ...m, [t]: true }));
  }, []);

  const changeTab = useCallback((t: PlaygroundTab) => {
    setTab(t);
    setMounted((m) => ({ ...m, [t]: true }));
    const url = new URL(window.location.href);
    if (t === 'math') url.searchParams.delete('tab');
    else url.searchParams.set('tab', t);
    window.history.replaceState(null, '', url.toString());
  }, []);

  async function toggleRecording() {
    if (isRecording) {
      mediaRecorder?.stop();
      setIsRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: false,
      });
      const localChunks: BlobPart[] = [];
      const mr = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) localChunks.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(localChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `alkahest-demo-${Date.now()}.webm`;
        a.click();
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start(1000);
      setMediaRecorder(mr);
      setIsRecording(true);
    } catch {
      alert('Screen capture not available or permission denied.');
    }
  }

  const confirmLeave = useCallback(
    () =>
      !isDirty ||
      window.confirm('Leave the playground? Unsaved changes will be lost.\n\nTip: press Ctrl+S (or ⌘S) to save first.'),
    [isDirty],
  );

  const markReady = useCallback(() => {
    document.documentElement.setAttribute('data-recording-ready', 'true');
  }, []);

  return (
    <>
      {!zenMode && (
        <>
          <SiteHeader confirmLeave={confirmLeave} />
          <PlaygroundBar
            tab={tab ?? 'math'}
            onTabChange={changeTab}
            serverStatus={tab === 'math' ? mathStatus : pythonStatus}
            serverLabel={tab === 'math' ? mathLabel : undefined}
            isRecording={isRecording}
            onToggleRecording={toggleRecording}
          />
        </>
      )}
      <main>
        <section id="panel-math" role="tabpanel" aria-labelledby="tab-math" hidden={tab !== 'math'}>
          {mounted.math && <MathNotebook onServerStatusChange={onMathStatus} onReady={markReady} />}
        </section>
        <section id="panel-python" role="tabpanel" aria-labelledby="tab-python" hidden={tab !== 'python'}>
          {mounted.python && (
            <>
              {!zenMode && <HostedBanner />}
              <Notebook
                zenMode={zenMode}
                onServerStatusChange={setPythonStatus}
                onDirtyChange={setIsDirty}
                onReady={markReady}
              />
            </>
          )}
        </section>
      </main>
    </>
  );
}
