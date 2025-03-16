import React, { ReactNode } from 'react';

interface PanelProps {
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}

const Panel = ({ children, className = '', noPadding = false }: PanelProps) => {
  return (
    <div
      className={`h-full overflow-auto bg-white rounded-lg shadow border ${
        noPadding ? '' : 'p-6'
      } ${className}`}
    >
      {children}
    </div>
  );
};

export default Panel;
