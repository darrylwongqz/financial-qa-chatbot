'use client';

import React from 'react';
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from './Accordion';
import {
  FileTextIcon,
  PercentIcon,
  CodeIcon,
  HelpCircleIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ContextItem {
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

interface ContextAccordionProps {
  contexts: ContextItem[];
  className?: string;
}

const ContextAccordion: React.FC<ContextAccordionProps> = ({
  contexts,
  className,
}) => {
  if (!contexts || contexts.length === 0) {
    return null;
  }

  return (
    <div className={cn('mt-4', className)}>
      <Accordion type="single" collapsible className="w-full">
        {contexts.map((context) => (
          <AccordionItem
            key={context.id}
            value={context.id}
            className="border border-gray-200 rounded-md mb-2 overflow-hidden"
          >
            <AccordionTrigger className="px-4 py-3 bg-gray-50 hover:bg-gray-100">
              <div className="flex items-center w-full">
                <FileTextIcon className="h-4 w-4 mr-2 text-gray-500 flex-shrink-0" />
                <div className="flex flex-col items-start flex-grow min-w-0">
                  <span className="text-sm font-medium truncate w-full">
                    {context.metadata.filename || 'Document'}
                    {context.metadata.document_type === 'conversation_turn' &&
                      ` (Q${context.metadata.turn_index + 1})`}
                  </span>
                  <div className="flex items-center mt-1">
                    <PercentIcon className="h-3 w-3 mr-1 text-indigo-500" />
                    <span className="text-xs text-gray-500">
                      Relevance: {Math.round(context.score * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-4 py-3 bg-white">
              <div className="space-y-3">
                {context.metadata.question && (
                  <div className="flex">
                    <HelpCircleIcon className="h-4 w-4 mr-2 text-indigo-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="text-xs font-medium text-gray-500 block">
                        Question:
                      </span>
                      <p className="text-sm">{context.metadata.question}</p>
                    </div>
                  </div>
                )}
                {context.metadata.answer !== undefined && (
                  <div className="flex">
                    <div className="h-4 w-4 mr-2 flex-shrink-0 mt-0.5 flex items-center justify-center">
                      <span className="text-xs font-bold text-indigo-500">
                        A:
                      </span>
                    </div>
                    <div>
                      <span className="text-xs font-medium text-gray-500 block">
                        Answer:
                      </span>
                      <p className="text-sm font-medium">
                        {context.metadata.answer}
                      </p>
                    </div>
                  </div>
                )}
                <div className="flex">
                  <FileTextIcon className="h-4 w-4 mr-2 text-indigo-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-xs font-medium text-gray-500 block">
                      Context:
                    </span>
                    <p className="text-sm whitespace-pre-wrap bg-gray-50 p-2 rounded-md mt-1">
                      {context.text}
                    </p>
                  </div>
                </div>
                {context.metadata.program && (
                  <div className="flex">
                    <CodeIcon className="h-4 w-4 mr-2 text-indigo-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="text-xs font-medium text-gray-500 block">
                        Calculation:
                      </span>
                      <p className="text-sm font-mono bg-gray-50 p-2 rounded-md mt-1">
                        {context.metadata.program}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
};

export default ContextAccordion;
