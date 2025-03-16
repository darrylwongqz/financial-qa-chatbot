import React from 'react';
import {
  BookOpenIcon,
  LightbulbIcon,
  MessageSquareTextIcon,
} from 'lucide-react';
import Panel from './ui/Panel';
import SectionTitle from './ui/SectionTitle';
import InfoCard from './ui/InfoCard';
import ExampleList from './ui/ExampleList';

const exampleQuestions = [
  'What was the revenue for Q3 2023?',
  'How does the current profit margin compare to last year?',
  'What is the debt-to-equity ratio?',
  'Explain the change in operating expenses from 2022 to 2023.',
  'What factors contributed to the increase in net income?',
];

const tips = [
  "Be specific about the time period you're asking about.",
  'For comparisons, clearly state what you want to compare.',
  'You can ask follow-up questions to get more details.',
  'The chatbot maintains context throughout your conversation.',
];

const InstructionsPanel = () => {
  return (
    <Panel>
      <SectionTitle>Financial Q&A Instructions</SectionTitle>

      <div className="space-y-6">
        <InfoCard icon={BookOpenIcon} title="About This Chatbot" highlighted>
          <p>
            This AI assistant is trained on the ConvFinQA dataset and can answer
            questions about financial statements, reports, and metrics. It can
            help you understand financial data and provide insights on company
            performance.
          </p>
        </InfoCard>

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
