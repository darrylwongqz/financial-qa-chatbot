import React, { ReactNode } from 'react';

interface SectionTitleProps {
  children: ReactNode;
  className?: string;
}

const SectionTitle = ({ children, className = '' }: SectionTitleProps) => {
  return (
    <h2 className={`text-xl font-bold mb-4 text-blue-600 ${className}`}>
      {children}
    </h2>
  );
};

export default SectionTitle;
