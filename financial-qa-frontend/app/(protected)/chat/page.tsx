'use client';

import React from 'react';
import InstructionsPanel from '@/components/InstructionsPanel';
import ChatInterface from '@/components/ChatInterface';
import ResizablePanelContainer from '@/components/ui/ResizablePanelContainer';

function ChatPage() {
  return (
    <div className="container mx-auto p-4 h-[calc(100vh-4rem)]">
      <ResizablePanelContainer
        leftPanel={<InstructionsPanel />}
        rightPanel={<ChatInterface />}
        initialLeftWidth={30}
        minLeftWidth={20}
        maxLeftWidth={50}
      />
    </div>
  );
}

export default ChatPage;
