'use client';

import React from 'react';
import resultsData from '../../../evaluation-results/results.json';
import { motion } from 'framer-motion';

// Define types for the evaluation data
type QuestionTypeMetrics = {
  total: number;
  error_or_no_context: {
    count: number;
    percentage: number;
  };
  successful: {
    count: number;
    has_calculation_steps?: number;
  };
};

type MetricAverage = {
  total: number;
  count: number;
  average: number;
};

type EvaluationResult = {
  id: string;
  status: string;
  retrieval_profile: 'fast' | 'balanced' | 'accurate';
  model: 'gpt-3.5-turbo' | 'gpt-4';
  created_at: string;
  updated_at: string;
  user_id: string;
  metrics: {
    total_count: number;
    error_count: number;
    error_rate: number;
    question_type_counts: {
      extraction: QuestionTypeMetrics;
      other: QuestionTypeMetrics;
      calculation: QuestionTypeMetrics;
      [key: string]: QuestionTypeMetrics; // Index signature for dynamic access
    };
    non_error_metrics: {
      total_count: number;
      numerical_accuracy: MetricAverage;
      financial_accuracy: MetricAverage;
      answer_relevance: MetricAverage;
      partial_numerical_match: MetricAverage;
      has_citations: MetricAverage;
      has_calculation_steps: MetricAverage;
    };
  };
};

// Helper function to format percentages
const formatPercentage = (value: number) => {
  return `${(value * 100).toFixed(1)}%`;
};

// Type assertion for the imported data
const typedResultsData = resultsData as EvaluationResult[];

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5 },
  },
};

const barVariants = {
  hidden: { width: 0 },
  visible: (width: number) => ({
    width: `${width}%`,
    transition: { duration: 0.8, ease: 'easeOut' },
  }),
};

function DashboardPage() {
  return (
    <div className="container mx-auto p-6">
      {/* Evaluation Banner */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-indigo-50 border-l-4 border-indigo-500 p-4 mb-6 rounded-md"
      >
        <div className="flex">
          <div className="flex-shrink-0">
            <svg
              className="h-5 w-5 text-indigo-500"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-indigo-700">
              This dashboard presents the evaluation results for our Financial
              QA Chatbot. The data shows performance metrics across different
              models and retrieval profiles.
            </p>
          </div>
        </div>
      </motion.div>

      <motion.h1
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="text-3xl font-bold mb-2"
      >
        Financial QA Chatbot Evaluation
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="text-gray-600 mb-6"
      >
        Comprehensive performance analysis across different configurations
      </motion.p>

      {/* Evaluation Scope Preamble */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="bg-white p-4 rounded-lg shadow-sm mb-6"
      >
        <h2 className="text-lg font-semibold mb-2">Evaluation Scope</h2>
        <p className="text-sm text-gray-700 mb-2">
          Our evaluation was conducted on a curated set of{' '}
          <strong>200 financial questions</strong> from the ConvFinQA dataset:
        </p>
        <div className="flex flex-wrap gap-2">
          <span className="inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium bg-purple-100 text-purple-800">
            <svg
              className="mr-1.5 h-2 w-2 text-purple-400"
              fill="currentColor"
              viewBox="0 0 8 8"
            >
              <circle cx="4" cy="4" r="3" />
            </svg>
            Calculation:{' '}
            {typedResultsData[0].metrics.question_type_counts.calculation.total}{' '}
            questions
          </span>
          <span className="inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
            <svg
              className="mr-1.5 h-2 w-2 text-blue-400"
              fill="currentColor"
              viewBox="0 0 8 8"
            >
              <circle cx="4" cy="4" r="3" />
            </svg>
            Extraction:{' '}
            {typedResultsData[0].metrics.question_type_counts.extraction.total}{' '}
            questions
          </span>
          <span className="inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium bg-amber-100 text-amber-800">
            <svg
              className="mr-1.5 h-2 w-2 text-amber-400"
              fill="currentColor"
              viewBox="0 0 8 8"
            >
              <circle cx="4" cy="4" r="3" />
            </svg>
            Other:{' '}
            {typedResultsData[0].metrics.question_type_counts.other.total}{' '}
            questions
          </span>
        </div>
      </motion.div>

      {/* Summary Cards */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
      >
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-lg shadow p-6 border-l-4 border-green-500"
        >
          <h2 className="text-lg font-semibold mb-2">Best Configuration</h2>
          <p className="text-2xl font-bold text-green-600">GPT-4 + Accurate</p>
          <p className="text-sm text-gray-500">
            Highest accuracy:{' '}
            {formatPercentage(
              typedResultsData[5].metrics.non_error_metrics.numerical_accuracy
                .average
            )}
          </p>
          <p className="text-sm text-gray-500">
            Error rate:{' '}
            {formatPercentage(typedResultsData[5].metrics.error_rate)}
          </p>
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500"
        >
          <h2 className="text-lg font-semibold mb-2">Recommended Default</h2>
          <p className="text-2xl font-bold text-blue-600">GPT-4 + Balanced</p>
          <p className="text-sm text-gray-500">
            Good accuracy:{' '}
            {formatPercentage(
              typedResultsData[4].metrics.non_error_metrics.numerical_accuracy
                .average
            )}
          </p>
          <p className="text-sm text-gray-500">
            Error rate:{' '}
            {formatPercentage(typedResultsData[4].metrics.error_rate)}
          </p>
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="bg-white rounded-lg shadow p-6 border-l-4 border-amber-500"
        >
          <h2 className="text-lg font-semibold mb-2">Budget Option</h2>
          <p className="text-2xl font-bold text-amber-600">
            GPT-3.5 + Accurate
          </p>
          <p className="text-sm text-gray-500">
            Decent accuracy:{' '}
            {formatPercentage(
              typedResultsData[2].metrics.non_error_metrics.numerical_accuracy
                .average
            )}
          </p>
          <p className="text-sm text-gray-500">
            Error rate:{' '}
            {formatPercentage(typedResultsData[2].metrics.error_rate)}
          </p>
        </motion.div>
      </motion.div>

      {/* Detailed Results Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="bg-white rounded-lg shadow mb-8"
      >
        <div className="px-6 py-4 border-b">
          <h2 className="text-xl font-semibold">Evaluation Results</h2>
          <p className="text-sm text-gray-500">
            Comparison across models and retrieval profiles
          </p>
          <div className="mt-2 text-xs text-amber-700 flex items-center">
            <svg
              className="h-4 w-4 mr-1 text-amber-500"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            Note: Accuracy metrics are calculated only on non-error responses,
            excluding failed retrievals.
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Configuration
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Error Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Numerical Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Financial Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Answer Relevance
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Has Citations
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Calculation Steps
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {typedResultsData.map((result, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="font-medium text-gray-900">
                      {result.model}
                    </div>
                    <div className="text-sm text-gray-500">
                      {result.retrieval_profile}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        result.metrics.error_rate < 0.1
                          ? 'bg-green-100 text-green-800'
                          : result.metrics.error_rate < 0.2
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {formatPercentage(result.metrics.error_rate)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {formatPercentage(
                      result.metrics.non_error_metrics.numerical_accuracy
                        .average
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {formatPercentage(
                      result.metrics.non_error_metrics.financial_accuracy
                        .average
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {formatPercentage(
                      result.metrics.non_error_metrics.answer_relevance.average
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {formatPercentage(
                      result.metrics.non_error_metrics.has_citations.average
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {formatPercentage(
                      (result.metrics.question_type_counts.calculation
                        .successful.has_calculation_steps || 0) /
                        (result.metrics.question_type_counts.calculation
                          .successful.count || 1)
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Question Type Analysis */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
      >
        {['extraction', 'calculation', 'other'].map((type) => (
          <motion.div
            key={type}
            variants={itemVariants}
            className="bg-white rounded-lg shadow p-6"
          >
            <h2 className="text-lg font-semibold mb-2 capitalize">
              {type} Questions{' '}
              <span className="text-sm font-normal text-gray-500">
                ({typedResultsData[0].metrics.question_type_counts[type].total}{' '}
                questions)
              </span>
            </h2>
            <div className="space-y-4">
              {typedResultsData
                .filter((_, i) => i % 3 === 0)
                .map((result, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center"
                  >
                    <span className="text-sm">
                      {result.model} ({result.retrieval_profile})
                    </span>
                    <div className="flex items-center">
                      <div className="w-32 bg-gray-200 rounded-full h-2.5">
                        <motion.div
                          initial="hidden"
                          animate="visible"
                          custom={
                            (1 -
                              result.metrics.question_type_counts[type]
                                .error_or_no_context.percentage) *
                            100
                          }
                          variants={barVariants}
                          className={`h-2.5 rounded-full ${
                            type === 'extraction'
                              ? 'bg-blue-600'
                              : type === 'calculation'
                              ? 'bg-purple-600'
                              : 'bg-amber-600'
                          }`}
                        ></motion.div>
                      </div>
                      <span className="ml-2 text-xs text-gray-600">
                        {formatPercentage(
                          1 -
                            result.metrics.question_type_counts[type]
                              .error_or_no_context.percentage
                        )}
                      </span>
                    </div>
                  </div>
                ))}
            </div>
            <p className="text-xs text-gray-500 mt-4">
              Success rate for {type} questions across configurations
            </p>
          </motion.div>
        ))}
      </motion.div>

      {/* Visual Comparison Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
        className="bg-white rounded-lg shadow mb-8"
      >
        <div className="px-6 py-4 border-b">
          <h2 className="text-xl font-semibold">Performance Comparison</h2>
          <p className="text-sm text-gray-500">
            Visual comparison of key metrics across configurations
          </p>
        </div>
        <div className="p-6">
          <div className="space-y-8">
            {/* Numerical Accuracy Chart */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-3">
                Numerical Accuracy
              </h3>
              <div className="space-y-4">
                {typedResultsData.map((result, index) => (
                  <div key={index} className="flex items-center">
                    <div className="w-32 text-sm text-right pr-4">
                      <span className="font-medium">{result.model}</span>
                      <span className="text-xs text-gray-500 block">
                        {result.retrieval_profile}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="w-full bg-gray-200 rounded-full h-4">
                        <motion.div
                          initial="hidden"
                          animate="visible"
                          custom={
                            result.metrics.non_error_metrics.numerical_accuracy
                              .average * 100
                          }
                          variants={barVariants}
                          className={`h-4 rounded-full ${
                            result.model === 'gpt-4'
                              ? 'bg-indigo-600'
                              : 'bg-amber-500'
                          }`}
                        ></motion.div>
                      </div>
                    </div>
                    <div className="w-16 text-sm font-medium text-gray-900 pl-4">
                      {formatPercentage(
                        result.metrics.non_error_metrics.numerical_accuracy
                          .average
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Error Rate Chart */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-3">
                Error Rate (lower is better)
              </h3>
              <div className="space-y-4">
                {typedResultsData.map((result, index) => (
                  <div key={index} className="flex items-center">
                    <div className="w-32 text-sm text-right pr-4">
                      <span className="font-medium">{result.model}</span>
                      <span className="text-xs text-gray-500 block">
                        {result.retrieval_profile}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="w-full bg-gray-200 rounded-full h-4">
                        <motion.div
                          initial="hidden"
                          animate="visible"
                          custom={result.metrics.error_rate * 100}
                          variants={barVariants}
                          className={`h-4 rounded-full ${
                            result.metrics.error_rate < 0.1
                              ? 'bg-green-500'
                              : result.metrics.error_rate < 0.2
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                          }`}
                        ></motion.div>
                      </div>
                    </div>
                    <div className="w-16 text-sm font-medium text-gray-900 pl-4">
                      {formatPercentage(result.metrics.error_rate)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Key Findings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
        className="bg-white rounded-lg shadow mb-8"
      >
        <div className="px-6 py-4 border-b">
          <h2 className="text-xl font-semibold">Key Findings</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-lg mb-3">Model Comparison</h3>
              <ul className="list-disc pl-5 space-y-2 text-sm">
                <li>
                  GPT-4 consistently outperforms GPT-3.5-turbo across all
                  metrics
                </li>
                <li>
                  GPT-4 with accurate retrieval achieves{' '}
                  {formatPercentage(
                    typedResultsData[5].metrics.non_error_metrics
                      .numerical_accuracy.average
                  )}{' '}
                  numerical accuracy vs{' '}
                  {formatPercentage(
                    typedResultsData[2].metrics.non_error_metrics
                      .numerical_accuracy.average
                  )}{' '}
                  for GPT-3.5-turbo
                </li>
                <li>
                  GPT-4 delivers more relevant answers (
                  {formatPercentage(
                    typedResultsData[5].metrics.non_error_metrics
                      .answer_relevance.average
                  )}{' '}
                  relevance with accurate retrieval)
                </li>
                <li>
                  Error rates are lower with GPT-4 (
                  {formatPercentage(typedResultsData[5].metrics.error_rate)}{' '}
                  with accurate retrieval) compared to GPT-3.5-turbo (
                  {formatPercentage(typedResultsData[2].metrics.error_rate)})
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-lg mb-3">
                Retrieval Profile Impact
              </h3>
              <ul className="list-disc pl-5 space-y-2 text-sm">
                <li>
                  <strong>Accurate:</strong> Lowest error rates, highest
                  accuracy, but slowest responses
                </li>
                <li>
                  <strong>Balanced:</strong> Good compromise between speed and
                  accuracy, recommended default
                </li>
                <li>
                  <strong>Fast:</strong> Highest error rates, lowest accuracy,
                  but fastest responses
                </li>
              </ul>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Recommendations */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bg-white rounded-lg shadow"
      >
        <div className="px-6 py-4 border-b">
          <h2 className="text-xl font-semibold">Recommendations</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div
              variants={itemVariants}
              className="bg-green-50 p-4 rounded-lg border border-green-200"
            >
              <h3 className="font-semibold text-green-800 mb-2">
                High-Precision Applications
              </h3>
              <p className="text-sm text-green-700 mb-2">
                <strong>Model:</strong> GPT-4
              </p>
              <p className="text-sm text-green-700 mb-2">
                <strong>Retrieval:</strong> Accurate
              </p>
              <p className="text-sm text-green-700">
                Delivers the highest numerical and financial accuracy, minimal
                error rate, and strong calculation transparency.
              </p>
            </motion.div>
            <motion.div
              variants={itemVariants}
              className="bg-blue-50 p-4 rounded-lg border border-blue-200"
            >
              <h3 className="font-semibold text-blue-800 mb-2">
                Most Use Cases (Recommended)
              </h3>
              <p className="text-sm text-blue-700 mb-2">
                <strong>Model:</strong> GPT-4
              </p>
              <p className="text-sm text-blue-700 mb-2">
                <strong>Retrieval:</strong> Balanced
              </p>
              <p className="text-sm text-blue-700">
                Offers performance metrics surprisingly close to accurate
                retrieval while providing significantly faster response times.
              </p>
            </motion.div>
            <motion.div
              variants={itemVariants}
              className="bg-amber-50 p-4 rounded-lg border border-amber-200"
            >
              <h3 className="font-semibold text-amber-800 mb-2">
                Speed-Critical Applications
              </h3>
              <p className="text-sm text-amber-700 mb-2">
                <strong>Model:</strong> GPT-3.5-turbo
              </p>
              <p className="text-sm text-amber-700 mb-2">
                <strong>Retrieval:</strong> Fast
              </p>
              <p className="text-sm text-amber-700">
                Although error rates are higher, it provides faster response
                times for real-time applications.
              </p>
            </motion.div>
            <motion.div
              variants={itemVariants}
              className="bg-purple-50 p-4 rounded-lg border border-purple-200"
            >
              <h3 className="font-semibold text-purple-800 mb-2">
                Budget-Constrained Scenarios
              </h3>
              <p className="text-sm text-purple-700 mb-2">
                <strong>Model:</strong> GPT-3.5-turbo
              </p>
              <p className="text-sm text-purple-700 mb-2">
                <strong>Retrieval:</strong> Accurate
              </p>
              <p className="text-sm text-purple-700">
                Prioritizes high-quality retrieval over model size when cost is
                a factor.
              </p>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Methodology Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
        className="mt-8 bg-white rounded-lg shadow"
      >
        <div className="px-6 py-4 border-b">
          <h2 className="text-xl font-semibold">Evaluation Methodology</h2>
        </div>
        <div className="p-6">
          <p className="text-sm text-gray-600 mb-4">
            Our evaluation framework assesses the chatbot using the following
            key metrics:
          </p>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.85 }}
            className="mb-4 bg-blue-50 p-3 border border-blue-200 rounded-md text-sm text-blue-800"
          >
            <strong>Important Note:</strong> Performance metrics (Numerical
            Accuracy, Financial Accuracy, Answer Relevance, etc.) are calculated
            only on non-error responses. This means these metrics represent the
            quality of successful responses, excluding cases where the model
            failed to retrieve sufficient context or provide a relevant answer.
            The Error Rate metric should be considered alongside performance
            metrics for a complete picture.
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.9 }}
              className="border rounded p-3"
            >
              <h3 className="font-medium text-gray-900">Error Rate</h3>
              <p className="text-xs text-gray-600">
                Percentage of questions where the model failed to provide a
                relevant answer or encountered context retrieval issues.
                Identified using pattern matching against phrases like
                &ldquo;I&apos;m sorry,&rdquo; &ldquo;couldn&apos;t find,&rdquo;
                etc.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.95 }}
              className="border rounded p-3"
            >
              <h3 className="font-medium text-gray-900">Numerical Accuracy</h3>
              <p className="text-xs text-gray-600">
                Binary measure (1 or 0) indicating whether numerical values
                match the ground truth within a tolerance of 1%. Values are
                normalized by removing currency symbols, converting text
                multipliers, etc.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 1.0 }}
              className="border rounded p-3"
            >
              <h3 className="font-medium text-gray-900">Financial Accuracy</h3>
              <p className="text-xs text-gray-600">
                Similar to numerical accuracy but tracked separately for
                financial figures. Currently uses the same 1% tolerance but
                allows for future adjustments.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 1.05 }}
              className="border rounded p-3"
            >
              <h3 className="font-medium text-gray-900">Answer Relevance</h3>
              <p className="text-xs text-gray-600">
                A score (0.0–1.0) based on answer length, presence of financial
                terms, and absence of error messages. Uses a heuristic approach
                with higher scores for substantive, financial term-rich
                responses.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 1.1 }}
              className="border rounded p-3"
            >
              <h3 className="font-medium text-gray-900">Has Citations</h3>
              <p className="text-xs text-gray-600">
                Binary indicator (1 or 0) showing whether the answer includes
                proper citations to sources. Detected using regex patterns for
                reference markers, attribution phrases, etc.
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 1.15 }}
              className="border rounded p-3"
            >
              <h3 className="font-medium text-gray-900">
                Has Calculation Steps
              </h3>
              <p className="text-xs text-gray-600">
                For calculation questions, indicates whether the model provides
                its calculation steps. Detected through mathematical operators,
                equals signs, and calculation-related terms.
              </p>
            </motion.div>
          </div>

          <div className="mt-8 p-4 bg-gray-50 rounded-md border border-gray-200">
            <h3 className="font-medium text-gray-900 mb-2">
              Methodology Limitations
            </h3>
            <ul className="text-xs text-gray-600 space-y-2 list-disc pl-4">
              <li>
                <strong>Sample Size:</strong> Evaluation used only 200 questions
                per configuration due to cost constraints.
              </li>
              <li>
                <strong>Evaluation Environment:</strong> Tests run on local
                hardware rather than production cloud environment.
              </li>
              <li>
                <strong>Question Distribution:</strong> Uneven distribution
                across question types may skew overall metrics.
              </li>
              <li>
                <strong>Binary Metrics:</strong> Some metrics use binary
                measures when reality is more nuanced.
              </li>
              <li>
                <strong>Non-Error Metrics:</strong> Performance metrics
                calculated only on successful responses may overstate
                performance.
              </li>
              <li>
                <strong>Automated Evaluation:</strong> Metrics rely on automated
                calculations rather than human judgment.
              </li>
              <li>
                <strong>Limited Retrieval Variations:</strong> Only three
                retrieval profiles were tested with fixed parameters.
              </li>
            </ul>
            <p className="text-xs text-gray-500 mt-2 italic">
              For complete methodology details, refer to our{' '}
              <a
                href="https://github.com/darrylwongqz/financial-qa-chatbot/blob/main/README_EVALUATION_RESULTS.md"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                documentation
              </a>
              .
            </p>
          </div>
        </div>
      </motion.div>

      {/* Conclusion Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1.0 }}
        className="mt-8 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg shadow text-white"
      >
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">Conclusion</h2>
          <p className="mb-6">
            Our Financial QA chatbot demonstrates robust performance across
            various configurations. GPT-4, paired with accurate retrieval,
            delivers the best results in terms of numerical precision,
            relevance, and calculation transparency. For most use cases, we
            recommend the balanced retrieval profile with GPT-4, which offers
            performance metrics surprisingly close to accurate retrieval while
            providing significantly faster response times.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <motion.a
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              href="/chat"
              className="inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-indigo-600 bg-white hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Try the Chatbot
            </motion.a>
            <motion.a
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              href="https://github.com/darrylwongqz/financial-qa-chatbot"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-700 bg-opacity-60 hover:bg-opacity-70 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              View Source Code
            </motion.a>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default DashboardPage;
