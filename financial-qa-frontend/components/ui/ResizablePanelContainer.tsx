import React, { ReactNode, useState } from 'react';

interface ResizablePanelContainerProps {
  leftPanel: ReactNode;
  rightPanel: ReactNode;
  initialLeftWidth?: number;
  minLeftWidth?: number;
  maxLeftWidth?: number;
  className?: string;
}

const ResizablePanelContainer = ({
  leftPanel,
  rightPanel,
  initialLeftWidth = 30,
  minLeftWidth = 20,
  maxLeftWidth = 50,
  className = '',
}: ResizablePanelContainerProps) => {
  const [leftWidth, setLeftWidth] = useState(initialLeftWidth);
  const [isResizing, setIsResizing] = useState(false);

  const handleMouseDown = () => {
    setIsResizing(true);
    document.body.style.cursor = 'col-resize';
  };

  const handleMouseUp = () => {
    if (isResizing) {
      setIsResizing(false);
      document.body.style.removeProperty('cursor');
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isResizing) return;

    const container = e.currentTarget as HTMLDivElement;
    const containerRect = container.getBoundingClientRect();
    const containerWidth = containerRect.width;

    const mouseX = e.clientX - containerRect.left;
    const newLeftWidthPercent = (mouseX / containerWidth) * 100;

    // Constrain to min/max values
    const clampedWidth = Math.max(
      minLeftWidth,
      Math.min(maxLeftWidth, newLeftWidthPercent)
    );

    setLeftWidth(clampedWidth);
  };

  React.useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isResizing) {
        setIsResizing(false);
        document.body.style.removeProperty('cursor');
      }
    };

    document.addEventListener('mouseup', handleGlobalMouseUp);
    return () => {
      document.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isResizing]);

  return (
    <div
      className={`flex h-full ${className}`}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <div
        className="h-full overflow-hidden"
        style={{ width: `${leftWidth}%` }}
      >
        {leftPanel}
      </div>

      <div
        className={`group w-1 bg-white hover:bg-blue-400 cursor-col-resize transition-all duration-200 ${
          isResizing ? 'bg-blue-500' : ''
        }`}
        onMouseDown={handleMouseDown}
      >
        <div className="h-full flex items-center justify-center overflow-visible">
          <div className="relative">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-0 opacity-0 group-hover:-translate-y-10 group-hover:opacity-100 transition-all duration-300 ease-out">
              <div className="w-[5px] h-[5px] rounded-full bg-blue-500 shadow-[0_0_3px_rgba(59,130,246,0.7)]"></div>
            </div>
            <div className="w-[5px] h-[5px] rounded-full bg-blue-500 shadow-[0_0_3px_rgba(59,130,246,0.7)] opacity-0 group-hover:opacity-100 transition-opacity duration-300 delay-75"></div>
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-0 opacity-0 group-hover:translate-y-10 group-hover:opacity-100 transition-all duration-300 ease-out">
              <div className="w-[5px] h-[5px] rounded-full bg-blue-500 shadow-[0_0_3px_rgba(59,130,246,0.7)]"></div>
            </div>
          </div>
        </div>
      </div>

      <div
        className="h-full overflow-hidden"
        style={{ width: `${100 - leftWidth - 0.25}%` }}
      >
        {rightPanel}
      </div>
    </div>
  );
};

export default ResizablePanelContainer;
