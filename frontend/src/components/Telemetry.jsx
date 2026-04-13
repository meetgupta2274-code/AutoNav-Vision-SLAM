import React from 'react';
import { Target, TrendingUp, Navigation, Zap } from 'lucide-react';

export default function Telemetry({ data }) {
  const metrics = [
    { label: 'Active Sector', value: data?.active_target || 'None', icon: Target, color: 'text-indigo-400', bg: 'bg-indigo-400/10' },
    { label: 'Track Confidence', value: data?.confidence ? `${(data.confidence * 100).toFixed(0)}%` : '0%', icon: Zap, color: 'text-emerald-400', bg: 'bg-emerald-400/10' },
    { label: 'Path Cells (ETA)', value: data?.path_len || '0', icon: Navigation, color: 'text-cyan-400', bg: 'bg-cyan-400/10' },
    { label: 'Proximity Score', value: data?.score ? data.score.toFixed(2) : '0.00', icon: TrendingUp, color: 'text-rose-400', bg: 'bg-rose-400/10' }
  ];

  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="glass-panel p-6">
        <h3 className="text-lg font-medium text-gray-300 mb-4 border-b border-white/10 pb-2">Systems Telemetry</h3>
        <div className="grid grid-cols-2 gap-4">
          {metrics.map((m, i) => (
            <div key={i} className="bg-white/5 p-4 rounded-xl border border-white/5 hover:bg-white/10 transition-colors">
              <div className="flex items-center gap-3 mb-2">
                <div className={`p-2 rounded-lg ${m.bg}`}>
                  <m.icon className={`w-5 h-5 ${m.color}`} />
                </div>
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{m.label}</span>
              </div>
              <div className="text-2xl font-bold tracking-tight mt-1">{m.value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-panel p-6 flex-grow flex flex-col">
        <h3 className="text-lg font-medium text-gray-300 mb-4 border-b border-white/10 pb-2">Analysis Log</h3>
        
        <div className="flex-grow flex items-center justify-center flex-col gap-3">
          <div className="relative w-48 h-48 flex items-center justify-center">
            {/* Pulsing rings for effect */}
            <div className="absolute inset-0 rounded-full border border-emerald-500/20 animate-[ping_3s_ease-out_infinite]"></div>
            <div className="absolute inset-4 rounded-full border border-emerald-500/30 animate-[ping_2s_ease-out_infinite] blur-[1px]"></div>
            <div className="absolute inset-8 rounded-full border border-emerald-500/40"></div>
            
            {/* Center target node */}
            <div className="z-10 text-center">
              <div className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-br from-emerald-400 to-cyan-500">
                {data?.targets_detected || 0}
              </div>
              <div className="text-xs text-gray-400 uppercase tracking-widest mt-1">Targets</div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 rounded-xl bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20">
          <div className="text-xs text-emerald-400 font-mono flex justify-between">
            <span>Nav Status</span>
            {data?.path_len > 0 ? (
              <span className="text-emerald-400 animate-pulse">OPTIMAL</span>
            ) : (
                <span className="text-rose-400">SEARCHING</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
