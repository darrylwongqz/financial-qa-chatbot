import React from 'react';

interface LoadingIndicatorProps {
  className?: string;
}

const LoadingIndicator = ({ className = '' }: LoadingIndicatorProps) => {
  return (
    <div className={`flex justify-start ${className}`}>
      <div className="bg-gray-100 rounded-lg p-4 shadow-sm max-w-[80%]">
        <div className="flex space-x-2">
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce" />
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce [animation-delay:0.2s]" />
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce [animation-delay:0.4s]" />
        </div>
      </div>
    </div>
  );
};

export default LoadingIndicator;
