import React, { ReactNode } from 'react';

type MessageRole = 'user' | 'assistant' | 'bot';

interface MessageBubbleProps {
  role: MessageRole;
  children: ReactNode;
  className?: string;
}

const MessageBubble = ({
  role,
  children,
  className = '',
}: MessageBubbleProps) => {
  // Treat 'bot' role the same as 'assistant'
  const isUser = role === 'user';

  return (
    <div
      className={`flex ${
        isUser ? 'justify-end' : 'justify-start'
      } ${className}`}
    >
      <div
        className={`max-w-[80%] rounded-lg p-4 shadow-sm ${
          isUser ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'
        }`}
      >
        <div className="whitespace-pre-wrap">{children}</div>
      </div>
    </div>
  );
};

export default MessageBubble;
