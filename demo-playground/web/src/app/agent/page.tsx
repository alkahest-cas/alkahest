'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import Nav from '@/components/ui/Nav';
import HostedBanner from '@/components/ui/HostedBanner';

function AgentChatLoading() {
  return (
    <div className="flex flex-col h-[calc(100vh-57px)]">
      <div className="flex-1" />
      <div className="border-t border-ak-border bg-ak-bg px-4 py-3">
        <div className="mx-auto max-w-3xl">
          <div className="rounded-lg border border-ak-border bg-ak-code-bg px-4 py-2.5 text-sm text-ak-muted">
            Loading agent…
          </div>
        </div>
      </div>
    </div>
  );
}

const AgentChat = dynamic(() => import('@/components/agent/AgentChat'), {
  ssr: false,
  loading: AgentChatLoading,
});

export default function AgentPage() {
  const [serverStatus, setServerStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');

  return (
    <>
      <Nav serverStatus={serverStatus} statusVariant="agent" />
      <HostedBanner variant="agent" />
      <AgentChat onServerStatusChange={setServerStatus} />
    </>
  );
}
