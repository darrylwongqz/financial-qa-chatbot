'use client';

import React from 'react';
import MessageBubble from './MessageBubble';

interface MessageWithContextProps {
  role: 'user' | 'assistant' | 'bot';
  message: string;
}

const MessageWithContext: React.FC<MessageWithContextProps> = ({
  role,
  message,
}) => {
  return <MessageBubble role={role}>{message}</MessageBubble>;
};

export default MessageWithContext;
