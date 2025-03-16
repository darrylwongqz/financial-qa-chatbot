import React from 'react';
import {
  BookOpenIcon,
  LightbulbIcon,
  MessageSquareTextIcon,
  FileTextIcon,
} from 'lucide-react';
import Panel from './ui/Panel';
import SectionTitle from './ui/SectionTitle';
import InfoCard from './ui/InfoCard';
import ExampleList from './ui/ExampleList';
import ContextAccordion, { ContextItem } from './ui/ContextAccordion';
import { useContextStore } from '@/lib/contextStore';

const exampleQuestions = [
  'What was the percentage change in the net cash from operating activities from 2008-2009?',
  'How did the S&P 500 perform in 2008?',
  'What percentage was that? (as a follow-up)',
];

const tips = [
  "Be specific about the time period you're asking about.",
  'For comparisons, clearly state what you want to compare.',
  'You can ask follow-up questions to get more details.',
  'The chatbot maintains context throughout your conversation.',
  "If you don't get the answer you need, try breaking down complex questions into simpler steps. Ask about available data first, then request calculations.",
];

interface ContextState {
  contexts: ContextItem[] | null;
}

const InstructionsPanel = () => {
  // Get contexts from the store
  const contexts = useContextStore((state: ContextState) => state.contexts);
  const hasContexts = contexts && contexts.length > 0;

  return (
    <Panel>
      <SectionTitle>Financial Q&A Instructions</SectionTitle>

      <div className="space-y-6">
        {hasContexts ? (
          <InfoCard icon={FileTextIcon} title="Sources & Context" highlighted>
            <div className="mb-3">
              <p className="text-sm text-gray-600 mb-2">
                The AI used the following sources to answer your question:
              </p>
              <ContextAccordion contexts={contexts} />
            </div>
          </InfoCard>
        ) : (
          <InfoCard icon={BookOpenIcon} title="About This Chatbot" highlighted>
            <p>
              This AI assistant is trained on the ConvFinQA dataset and can
              answer questions about financial statements, reports, and metrics.
              It can help you understand financial data and provide insights on
              company performance.
            </p>
          </InfoCard>
        )}

        <InfoCard icon={MessageSquareTextIcon} title="Example Questions">
          <ExampleList items={exampleQuestions} />
        </InfoCard>

        <InfoCard
          icon={LightbulbIcon}
          title="Tips for Better Results"
          highlighted
        >
          <ExampleList items={tips} />
        </InfoCard>
      </div>
    </Panel>
  );
};

export default InstructionsPanel;
