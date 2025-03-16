import React from 'react';

function DashboardPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Stats Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-2">Chat Statistics</h2>
          <p className="text-3xl font-bold text-indigo-600">0</p>
          <p className="text-sm text-gray-500">Total conversations</p>
        </div>

        {/* Recent Activity Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-2">Recent Activity</h2>
          <p className="text-gray-600">No recent activity</p>
        </div>

        {/* Settings Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-2">Settings</h2>
          <p className="text-gray-600 mb-4">Configure your preferences</p>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors">
            Manage Settings
          </button>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
