import React, { ReactNode } from 'react';
import { LucideIcon } from 'lucide-react';

interface InfoCardProps {
  icon: LucideIcon;
  title: string;
  children: ReactNode;
  highlighted?: boolean;
}

const InfoCard = ({
  icon: Icon,
  title,
  children,
  highlighted = false,
}: InfoCardProps) => {
  return (
    <div className={`p-4 rounded-lg ${highlighted ? 'bg-blue-50' : ''}`}>
      <div className="flex items-start mb-2">
        <Icon className="h-5 w-5 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
        <h3 className="font-semibold text-gray-800">{title}</h3>
      </div>
      <div className="text-gray-700 text-sm">{children}</div>
    </div>
  );
};

export default InfoCard;
