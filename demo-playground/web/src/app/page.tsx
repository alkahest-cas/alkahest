'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import Nav from '@/components/ui/Nav';
import HostedBanner from '@/components/ui/HostedBanner';
import { readZenFromUrl } from '@/lib/recording';

// Load Notebook client-side only (uses Web Worker + CodeMirror)
const Notebook = dynamic(() => import('@/components/notebook/Notebook'), { ssr: false });

export default function NotebookPage() {
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [serverStatus, setServerStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');
  const [zenMode] = useState(readZenFromUrl);

  async function toggleRecording() {
    if (isRecording) {
      mediaRecorder?.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: { frameRate: 30 },
          audio: false,
        });
        const localChunks: BlobPart[] = [];
        const mr = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
        mr.ondataavailable = (e) => { if (e.data.size > 0) localChunks.push(e.data); };
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
  }

  useEffect(() => {
    if (!zenMode) return;
    const markReady = () => document.documentElement.setAttribute('data-recording-ready', 'true');
    const editors = document.querySelectorAll('.cm-editor');
    if (editors.length > 0) markReady();
  }, [zenMode]);

  return (
    <>
      {!zenMode && (
        <Nav
          isRecording={isRecording}
          onToggleRecording={toggleRecording}
          serverStatus={serverStatus}
          zenMode={zenMode}
        />
      )}
      <main>
        {!zenMode && <HostedBanner />}
        <Notebook
          zenMode={zenMode}
          onServerStatusChange={setServerStatus}
          onReady={() => document.documentElement.setAttribute('data-recording-ready', 'true')}
        />
      </main>
    </>
  );
}
