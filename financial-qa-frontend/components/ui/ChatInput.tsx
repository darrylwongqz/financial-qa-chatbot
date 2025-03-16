import React from 'react';
import { Button } from '@/components/ui/button';
import { SendIcon } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isLoading: boolean;
  className?: string;
}

const ChatInput = ({
  value,
  onChange,
  onSend,
  onKeyDown,
  isLoading,
  className = '',
}: ChatInputProps) => {
  return (
    <div className={`border-t p-4 bg-gray-50 ${className}`}>
      <div className="flex items-center space-x-3">
        <div className="relative flex-1 flex items-center">
          <textarea
            className="w-full border border-gray-300 rounded-lg p-3 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent overflow-y-auto shadow-sm"
            placeholder="Ask a financial question..."
            rows={2}
            value={value}
            onChange={onChange}
            onKeyDown={onKeyDown}
            style={{
              height: '60px',
              lineHeight: '1.5',
              boxSizing: 'border-box',
            }}
          />
        </div>
        <div className="flex-shrink-0">
          <Button
            onClick={onSend}
            disabled={!value.trim() || isLoading}
            className="bg-blue-600 hover:bg-blue-700 text-white cursor-pointer h-[60px] w-[60px] p-0 flex items-center justify-center rounded-lg shadow-md transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:hover:scale-100"
          >
            <SendIcon className="h-5 w-5" />
          </Button>
        </div>
      </div>
      <div className="mt-2 text-xs text-gray-500 text-right">
        Press Enter to send, Shift+Enter for a new line
      </div>
    </div>
  );
};

export default ChatInput;
