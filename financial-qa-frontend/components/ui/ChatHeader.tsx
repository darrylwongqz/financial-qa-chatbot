import React from 'react';
import { Button } from '@/components/ui/button';
import {
  RefreshCwIcon,
  ZapIcon,
  ScaleIcon,
  SearchIcon,
  BrainIcon,
  SparklesIcon,
} from 'lucide-react';

// Define the RetrievalProfile type locally
type RetrievalProfile = 'fast' | 'balanced' | 'accurate';
// Define the Model type locally
type Model = 'gpt-3.5-turbo' | 'gpt-4';

interface ChatHeaderProps {
  title: string;
  mode: RetrievalProfile;
  onModeChange: (mode: RetrievalProfile) => void;
  model: Model;
  onModelChange: (model: Model) => void;
  onClear: () => void;
  className?: string;
}

const ChatHeader = ({
  title,
  mode,
  onModeChange,
  model,
  onModelChange,
  onClear,
  className = '',
}: ChatHeaderProps) => {
  return (
    <div
      className={`p-4 border-b flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 ${className}`}
    >
      <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2">
        {/* Retrieval Profile Selector */}
        <div className="flex items-center bg-gray-100 rounded-lg p-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onModeChange('fast')}
            className={`flex items-center px-2 py-1 text-xs rounded cursor-pointer ${
              mode === 'fast'
                ? 'bg-white shadow-sm text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <ZapIcon className="h-3 w-3 mr-1" />
            Fast
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onModeChange('balanced')}
            className={`flex items-center px-2 py-1 text-xs rounded cursor-pointer ${
              mode === 'balanced'
                ? 'bg-white shadow-sm text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <ScaleIcon className="h-3 w-3 mr-1" />
            Balanced
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onModeChange('accurate')}
            className={`flex items-center px-2 py-1 text-xs rounded cursor-pointer ${
              mode === 'accurate'
                ? 'bg-white shadow-sm text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <SearchIcon className="h-3 w-3 mr-1" />
            Accurate
          </Button>
        </div>

        {/* Model Selector */}
        <div className="flex items-center bg-gray-100 rounded-lg p-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onModelChange('gpt-3.5-turbo')}
            className={`flex items-center px-2 py-1 text-xs rounded cursor-pointer ${
              model === 'gpt-3.5-turbo'
                ? 'bg-white shadow-sm text-green-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <BrainIcon className="h-3 w-3 mr-1" />
            GPT-3.5
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onModelChange('gpt-4')}
            className={`flex items-center px-2 py-1 text-xs rounded cursor-pointer ${
              model === 'gpt-4'
                ? 'bg-white shadow-sm text-purple-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <SparklesIcon className="h-3 w-3 mr-1" />
            GPT-4
          </Button>
        </div>

        {/* Clear Button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onClear}
          className="text-gray-500 hover:text-gray-700 cursor-pointer"
        >
          <RefreshCwIcon className="h-4 w-4 mr-1" />
          Clear
        </Button>
      </div>
    </div>
  );
};

export default ChatHeader;
