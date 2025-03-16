'use client';

import { Button } from '@/components/ui/button';
import {
  BrainCogIcon,
  DatabaseIcon,
  GaugeIcon,
  MessageSquareTextIcon,
  SearchIcon,
  ZapIcon,
} from 'lucide-react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { useUser } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';

const features = [
  {
    name: 'Vector-Powered Knowledge Retrieval',
    description:
      'Advanced Retrieval Augmented Generation with Pinecone vector embeddings delivers precise answers from vast financial datasets.',
    icon: DatabaseIcon,
  },
  {
    name: 'Blazing Fast Responses',
    description:
      'Instant AI-powered answers that transform complex financial questions into clear, actionable insights in seconds.',
    icon: ZapIcon,
  },
  {
    name: 'Conversation Memory',
    description:
      'Smart conversations that maintain context across your entire session, creating truly coherent financial discussions.',
    icon: BrainCogIcon,
  },
  {
    name: 'Customizable Intelligence Modes',
    description:
      'Choose your perfect balance with three intelligence modes: Fast for quick answers, Balanced for everyday use, or Accurate for in-depth analysis.',
    icon: GaugeIcon,
  },
  {
    name: 'Comprehensive Financial Knowledge',
    description:
      'Trained on the extensive ConvFinQA dataset to provide expert-level insights on financial statements, reports, and metrics.',
    icon: SearchIcon,
  },
  {
    name: 'Contextual Understanding',
    description:
      'Natural conversation flow that understands the nuances of financial terminology and maintains context throughout your dialogue.',
    icon: MessageSquareTextIcon,
  },
];

// Animation variants
const fadeIn = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.6 } },
};

const slideUp = {
  hidden: { y: 30, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { duration: 0.6 } },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.3,
    },
  },
};

const featureItem = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 50,
      damping: 10,
    },
  },
};

const chatBubbleAnimation = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 50,
      damping: 10,
    },
  },
};

export default function Home() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });

  const backgroundY = useTransform(scrollYProgress, [0, 1], ['0%', '30%']);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [1, 0.8, 0.6]);
  const { isSignedIn } = useUser();
  const router = useRouter();

  const handleStartChatting = () => {
    if (isSignedIn) {
      router.push('/chat');
    } else {
      router.push('/sign-in');
    }
  };

  return (
    <main className="flex-1 overflow-auto p-2 lg:p-5 bg-gradient-to-bl from-white to-blue-600">
      <motion.div
        ref={containerRef}
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        style={{ opacity }}
        className="landing-container py-24 sm:py-32"
      >
        {/* Add a subtle parallax background */}
        <motion.div
          className="absolute inset-0 -z-10 overflow-hidden rounded-md"
          style={{ y: backgroundY }}
        >
          <div className="absolute inset-0 bg-gradient-to-tr from-blue-50 to-cyan-50 opacity-50" />
          <div
            className="absolute inset-0 opacity-5"
            style={{
              backgroundImage:
                'radial-gradient(circle, rgba(0,0,0,0.1) 1px, transparent 1px)',
              backgroundSize: '20px 20px',
            }}
          />
        </motion.div>

        <div className="flex flex-col justify-center items-center mx-auto max-w-7xl px-6 lg:px-8">
          <motion.div
            variants={slideUp}
            className="mx-auto max-w-2xl sm:text-center"
          >
            <motion.h2
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="text-base font-semibold leading-7 text-blue-600"
            >
              Your Financial Intelligence Assistant
            </motion.h2>

            <motion.p
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.5 }}
              className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-6xl"
            >
              Transform Financial Data into Intelligent Conversations
            </motion.p>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.5 }}
              className="mt-6 text-lg leading-8 text-gray-600"
            >
              Introducing{' '}
              <span className="font-bold bg-gradient-to-r from-blue-600 to-cyan-600 text-transparent bg-clip-text">
                FinancialQA Chatbot
              </span>
              — the AI assistant that transforms complex financial information
              into <span className="font-bold">clear, actionable insights</span>
              , enhancing financial decision-making effortlessly.
            </motion.p>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8, duration: 0.5 }}
              className="mt-4 text-lg leading-8 text-gray-600"
            >
              Simply ask your financial question, and
              <span className="font-semibold italic">
                {' '}
                get expert-level answers instantly.
              </span>
            </motion.p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              delay: 1,
              duration: 0.5,
              type: 'spring',
              stiffness: 100,
            }}
          >
            <Button
              onClick={handleStartChatting}
              className="mt-10 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 transition-all duration-200 transform hover:scale-105 shadow-md hover:shadow-lg text-white font-medium px-10 py-6 text-lg rounded-md cursor-pointer"
            >
              Start Chatting
            </Button>
          </motion.div>
        </div>

        <div className="relative overflow-hidden pt-16">
          <div className="mx-auto max-w-7xl px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                delay: 0.5,
                duration: 0.8,
                type: 'spring',
                stiffness: 50,
              }}
              className="mb-[-0%] rounded-xl shadow-2xl ring-1 ring-gray-900/10 bg-gradient-to-r from-blue-100 to-cyan-100 h-[600px] flex items-center justify-center relative overflow-hidden"
              style={{ maxHeight: '80vh' }}
            >
              {/* Add subtle animated background elements */}
              <motion.div
                className="absolute w-64 h-64 rounded-full bg-blue-200/30 blur-3xl"
                animate={{
                  x: [0, 100, 0],
                  y: [0, 50, 0],
                }}
                transition={{
                  duration: 20,
                  repeat: Infinity,
                  repeatType: 'reverse',
                  ease: 'easeInOut',
                }}
                style={{ top: '10%', left: '10%' }}
              />
              <motion.div
                className="absolute w-72 h-72 rounded-full bg-cyan-200/30 blur-3xl"
                animate={{
                  x: [0, -80, 0],
                  y: [0, 40, 0],
                }}
                transition={{
                  duration: 15,
                  repeat: Infinity,
                  repeatType: 'reverse',
                  ease: 'easeInOut',
                }}
                style={{ bottom: '10%', right: '10%' }}
              />

              <motion.div
                animate={{
                  y: [0, -10, 0],
                }}
                transition={{
                  duration: 5,
                  repeat: Infinity,
                  repeatType: 'mirror',
                  ease: 'easeInOut',
                }}
                className="text-center p-8 bg-white/80 backdrop-blur-sm rounded-lg shadow-md z-10"
              >
                <motion.h3
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.2, duration: 0.5 }}
                  className="text-2xl font-bold text-blue-600 mb-2"
                >
                  Financial QA Chatbot
                </motion.h3>
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.4, duration: 0.5 }}
                  className="text-gray-700 mb-4"
                >
                  Intelligent financial insights powered by advanced AI
                </motion.p>
                <div className="flex flex-col space-y-3 text-left">
                  <motion.div
                    variants={chatBubbleAnimation}
                    initial="hidden"
                    animate="visible"
                    transition={{ delay: 1.6 }}
                    className="bg-blue-50 p-3 rounded-lg"
                  >
                    <p className="text-sm text-gray-600">
                      What was the net income for Q3 2023?
                    </p>
                  </motion.div>
                  <motion.div
                    variants={chatBubbleAnimation}
                    initial="hidden"
                    animate="visible"
                    transition={{ delay: 2.0 }}
                    className="bg-blue-100 p-3 rounded-lg ml-4"
                  >
                    <p className="text-sm text-gray-800">
                      The net income for Q3 2023 was $4.2 million, which
                      represents a 15% increase from the previous quarter.
                    </p>
                  </motion.div>
                  <motion.div
                    variants={chatBubbleAnimation}
                    initial="hidden"
                    animate="visible"
                    transition={{ delay: 2.4 }}
                    className="bg-blue-50 p-3 rounded-lg"
                  >
                    <p className="text-sm text-gray-600">
                      How does that compare to the same period last year?
                    </p>
                  </motion.div>
                  <motion.div
                    variants={chatBubbleAnimation}
                    initial="hidden"
                    animate="visible"
                    transition={{ delay: 2.8 }}
                    className="bg-blue-100 p-3 rounded-lg ml-4"
                  >
                    <p className="text-sm text-gray-800">
                      Compared to Q3 2022, this represents a 22% year-over-year
                      growth from $3.45 million, driven primarily by the
                      expansion of the company&apos;s digital services division.
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            </motion.div>
            <div aria-hidden="true" className="relative">
              <div className="absolute bottom-0 -inset-x-32 bg-gradient-to-t from-white/95 pt-[5%]" />
            </div>
          </div>
        </div>

        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="mx-auto mt-16 max-w-7xl px-6 sm:mt-20 md:mt-24 lg:px-8"
        >
          <dl className="mx-auto grid max-w-2xl grid-cols-1 gap-x-6 gap-y-10 text-base leading-7 text-gray-600 sm:grid-cols-2 lg:mx-0 lg:max-w-none lg:grid-cols-3 lg:gap-x-8 lg:gap-y-16">
            {features.map((feature, index) => (
              <motion.div
                key={feature.name}
                variants={featureItem}
                className="relative pl-9 hover:scale-105 transition-transform duration-300"
              >
                <dt className="inline font-semibold text-gray-900">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{
                      delay: 0.5 + index * 0.1,
                      type: 'spring',
                      stiffness: 100,
                    }}
                  >
                    <feature.icon
                      aria-hidden="true"
                      className="absolute left-1 top-1 h-5 w-5 text-blue-600"
                    />
                  </motion.div>
                  {feature.name}
                </dt>
                <dd className="mt-2">{feature.description}</dd>
              </motion.div>
            ))}
          </dl>
        </motion.div>
      </motion.div>
    </main>
  );
}
