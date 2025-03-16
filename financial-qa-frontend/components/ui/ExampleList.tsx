import React from 'react';

interface ExampleListProps {
  items: string[];
  className?: string;
}

const ExampleList = ({ items, className = '' }: ExampleListProps) => {
  return (
    <ul
      className={`space-y-1 text-sm text-gray-700 ml-7 list-disc ${className}`}
    >
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
};

export default ExampleList;
