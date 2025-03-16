'use client';

import React from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

interface ResizablePanelProps {
  leftPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  initialLeftWidth?: number; // percentage
  minLeftWidth?: number; // percentage
  maxLeftWidth?: number; // percentage
  onResizeStart?: () => void;
  onResizeEnd?: () => void;
  isResizing?: boolean;
}

const ResizablePanel: React.FC<ResizablePanelProps> = ({
  leftPanel,
  rightPanel,
  initialLeftWidth = 33,
  minLeftWidth = 20,
  maxLeftWidth = 50,
  onResizeStart,
  onResizeEnd,
  isResizing = false,
}) => {
  return (
    <PanelGroup
      direction="horizontal"
      className={`h-full rounded-lg overflow-hidden ${
        isResizing ? 'select-none' : ''
      }`}
      onLayout={onResizeEnd}
    >
      <Panel
        defaultSize={initialLeftWidth}
        minSize={minLeftWidth}
        maxSize={maxLeftWidth}
        className="overflow-hidden"
      >
        {leftPanel}
      </Panel>

      <PanelResizeHandle
        className="group relative mx-1 transition-all duration-200 ease-in-out hover:mx-0 cursor-col-resize"
        onDragging={(isDragging) => {
          if (isDragging && onResizeStart) onResizeStart();
          if (!isDragging && onResizeEnd) onResizeEnd();
        }}
      >
        {/* Custom resize handle */}
        <div className="absolute inset-0 flex items-center justify-center">
          {/* Vertical line */}
          <div className="w-0.5 h-full bg-gray-200 group-hover:bg-blue-400 transition-all duration-200"></div>

          {/* Middle handle with arrows */}
          <div className="absolute bg-gray-300 group-hover:bg-blue-500 h-20 px-1 rounded flex items-center justify-center shadow-md transition-all duration-200">
            {/* Double-headed arrow */}
            <div className="text-white font-bold text-xl">⟷</div>
          </div>
        </div>
      </PanelResizeHandle>
      <Panel className="overflow-hidden">{rightPanel}</Panel>
    </PanelGroup>
  );
};

export default ResizablePanel;
