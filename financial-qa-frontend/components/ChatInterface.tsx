'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import Panel from './ui/Panel';
import ChatHeader from './ui/ChatHeader';
import LoadingIndicator from './ui/LoadingIndicator';
import ChatInput from './ui/ChatInput';
import ConfirmDialog from './ui/ConfirmDialog';
import { useUser } from '@clerk/nextjs';
import { db } from '@/lib/firebase';
import { collection, query, orderBy } from 'firebase/firestore';
import { useCollection } from 'react-firebase-hooks/firestore';
import { useContextStore } from '@/lib/contextStore';
import MessageWithContext from './ui/MessageWithContext';
import { ContextItem } from './ui/ContextAccordion';

// Types
type MessageRole = 'user' | 'bot';
type RetrievalProfile = 'fast' | 'balanced' | 'accurate';
type Model = 'gpt-3.5-turbo' | 'gpt-4';

interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

interface Message {
  sender: MessageRole;
  message: string;
  conversation_id: string;
  timestamp: string;
  retrieval_profile: RetrievalProfile;
  model: Model | null;
  token_usage: TokenUsage | null;
}

interface ChatResponse {
  question: string;
  answer: string;
  model: Model;
  conversation_id: string;
  context: Context[];
  token_usage: TokenUsage;
  processing_time: number;
  retrieval_profile: RetrievalProfile;
}

export interface Context {
  id: string;
  text: string;
  metadata: {
    answer: number;
    cleaned: boolean;
    document_type: string;
    filename: string;
    is_split: boolean;
    part: number;
    program: string;
    question: string;
    source_id: string;
    text: string;
    text_length: number;
    total_parts: number;
    turn_index: number;
    word_count: number;
  };
  score: number;
}

interface ContextState {
  contexts: ContextItem[] | null;
  setContexts: (contexts: ContextItem[] | null) => void;
  clearContexts: () => void;
}

const ChatInterface = () => {
  const { user } = useUser();
  const userId = user?.primaryEmailAddress?.emailAddress;

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [retrievalProfile, setRetrievalProfile] =
    useState<RetrievalProfile>('balanced');
  const [selectedModel, setSelectedModel] = useState<Model>('gpt-4');
  const [currentConversationId, setCurrentConversationId] = useState<
    string | null
  >(null);

  // For optimistic UI updates
  const [optimisticMessages, setOptimisticMessages] = useState<Message[]>([]);
  const [waitingForBotResponse, setWaitingForBotResponse] = useState(false);

  // Use Firebase collection hook to subscribe to chat messages
  const [snapshot, loading] = useCollection(
    userId
      ? query(
          collection(db, 'users', userId, 'chat'),
          orderBy('timestamp', 'asc')
        )
      : null
  );

  // Convert snapshot to messages using useMemo
  const firestoreMessages = useMemo(() => {
    return snapshot
      ? snapshot.docs.map((doc) => {
          const data = doc.data() as Message;
          return {
            ...data,
            timestamp: data.timestamp || new Date().toISOString(),
          };
        })
      : [];
  }, [snapshot]);

  // Combine firestore messages with optimistic messages
  const messages = useMemo(() => {
    // If we have firestore messages, use those (they're the source of truth)
    if (firestoreMessages.length > 0) {
      return firestoreMessages;
    }

    // Otherwise, if we're not loading and have no messages, show welcome message
    if (
      !loading &&
      firestoreMessages.length === 0 &&
      optimisticMessages.length === 0 &&
      !waitingForBotResponse
    ) {
      return [
        {
          sender: 'bot' as MessageRole,
          message:
            "👋 Hi there! I'm Finance Bot, your AI assistant for financial questions. Start by asking me a question about financial data!",
          conversation_id: 'welcome',
          timestamp: new Date().toISOString(),
          retrieval_profile: 'balanced',
          model: null,
          token_usage: null,
        },
      ];
    }

    // If we have optimistic messages, use those
    return optimisticMessages;
  }, [firestoreMessages, optimisticMessages, loading, waitingForBotResponse]);

  // Find the most recent conversation ID
  useEffect(() => {
    if (firestoreMessages.length > 0) {
      const conversations = [
        ...new Set(firestoreMessages.map((msg) => msg.conversation_id)),
      ];
      if (conversations.length > 0) {
        setCurrentConversationId(conversations[conversations.length - 1]);
      }
    }
  }, [firestoreMessages]);

  // Ref to track the end of the messages
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to the last message whenever messages change
  useEffect(() => {
    if (endOfMessagesRef.current) {
      endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Get the context store
  const setContexts = useContextStore(
    (state: ContextState) => state.setContexts
  );
  const clearContexts = useContextStore(
    (state: ContextState) => state.clearContexts
  );

  const handleSendMessage = async () => {
    if (!input.trim() || !userId) return;

    const userQuestion = input;
    setInput(''); // Clear input immediately

    // Create optimistic user message
    const tempConversationId = currentConversationId || `temp-${Date.now()}`;
    const optimisticUserMsg: Message = {
      sender: 'user',
      message: userQuestion,
      conversation_id: tempConversationId,
      timestamp: new Date().toISOString(),
      retrieval_profile: retrievalProfile,
      model: null,
      token_usage: null,
    };

    // Update UI immediately with user message
    setOptimisticMessages((prevMessages) => [
      ...prevMessages,
      optimisticUserMsg,
    ]);

    // Then set loading states
    setIsLoading(true);
    setWaitingForBotResponse(true);

    try {
      const response = await fetch(`/api/users/${userId}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userQuestion,
          model: selectedModel,
          temperature: 0.7,
          max_tokens: 1000,
          retrieval_profile: retrievalProfile,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data: ChatResponse = await response.json();

      // Store the context data
      if (data.context && data.context.length > 0) {
        setContexts(data.context);
      }

      // Update conversation ID if this is a new conversation
      if (!currentConversationId) {
        setCurrentConversationId(data.conversation_id);
      }

      // Add bot response to optimistic messages
      const optimisticBotMsg: Message = {
        sender: 'bot',
        message: data.answer,
        conversation_id: data.conversation_id,
        timestamp: new Date().toISOString(),
        retrieval_profile: retrievalProfile,
        model: data.model,
        token_usage: data.token_usage,
      };

      setOptimisticMessages((prevMessages) => [
        ...prevMessages,
        optimisticBotMsg,
      ]);

      // Clear optimistic messages after a delay to let Firestore catch up
      setTimeout(() => {
        setOptimisticMessages([]);
        setWaitingForBotResponse(false);
      }, 300);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message (without duplicating the user message)
      const errorMsg: Message = {
        sender: 'bot',
        message:
          "I'm sorry, I couldn't process your request. Please try again.",
        conversation_id: tempConversationId,
        timestamp: new Date().toISOString(),
        retrieval_profile: retrievalProfile,
        model: null,
        token_usage: null,
      };

      setOptimisticMessages((prevMessages) => [...prevMessages, errorMsg]);
      setWaitingForBotResponse(false);

      // Clear contexts on error
      clearContexts();
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearButtonClick = () => {
    // Only show confirmation if there are messages to clear
    if (firestoreMessages.length > 0 || optimisticMessages.length > 0) {
      setShowClearConfirm(true);
    } else {
      // If no messages, just reset the state
      setCurrentConversationId(null);
    }
  };

  const clearChat = async () => {
    if (!userId) return;

    setIsDeleting(true);
    // Clear optimistic messages immediately for better UX
    setOptimisticMessages([]);
    setWaitingForBotResponse(false);

    try {
      const response = await fetch(`/api/users/${userId}/chat`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to clear chat');
      }

      setCurrentConversationId(null);

      // Add welcome message after clearing
      setOptimisticMessages([
        {
          sender: 'bot' as MessageRole,
          message: "Chat history cleared! I'm ready for new questions.",
          conversation_id: 'welcome',
          timestamp: new Date().toISOString(),
          retrieval_profile: 'balanced',
          model: null,
          token_usage: null,
        },
      ]);

      // Clear contexts when clearing chat
      clearContexts();
    } catch (error) {
      console.error('Error clearing chat:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Panel className="flex flex-col overflow-hidden" noPadding>
      <ChatHeader
        title="Financial Q&A Chat"
        mode={retrievalProfile}
        onModeChange={setRetrievalProfile}
        model={selectedModel}
        onModelChange={setSelectedModel}
        onClear={handleClearButtonClick}
      />

      {/* Messages container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {loading ? (
          <div className="flex h-full items-center justify-center">
            <LoadingIndicator />
          </div>
        ) : isDeleting ? (
          <div className="flex flex-col h-full items-center justify-center">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 border-4 border-t-blue-500 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"></div>
              <div className="absolute inset-2 border-4 border-t-transparent border-r-blue-300 border-b-transparent border-l-transparent rounded-full animate-spin-slow"></div>
              <div className="absolute inset-4 border-4 border-t-transparent border-r-transparent border-b-blue-200 border-l-transparent rounded-full animate-spin-slower"></div>
            </div>
            <p className="mt-4 text-gray-500">Clearing chat history...</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-gray-500">
            <p>No messages yet. Start by asking a financial question.</p>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <MessageWithContext
                key={index}
                role={message.sender}
                message={message.message}
              />
            ))}
            {/* Show loading indicator for bot response */}
            {waitingForBotResponse && (
              <div className="flex justify-start my-4 ml-2">
                <LoadingIndicator />
              </div>
            )}
            {/* Ref to ensure auto-scrolling to the last message */}
            <div ref={endOfMessagesRef} />
          </>
        )}
      </div>

      <ChatInput
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onSend={handleSendMessage}
        onKeyDown={handleKeyDown}
        isLoading={isLoading || isDeleting}
      />

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        onConfirm={clearChat}
        title="Clear Chat History"
        message="Are you sure you want to clear your chat history? The Financial Q&A bot will no longer have context from your previous conversations, which may affect the quality of future responses."
        confirmText="Clear History"
        cancelText="Cancel"
      />
    </Panel>
  );
};

export default ChatInterface;
