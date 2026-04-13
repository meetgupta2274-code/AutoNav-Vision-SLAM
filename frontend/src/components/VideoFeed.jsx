import { useEffect, useRef } from 'react';

export default function VideoFeed({ src }) {
  const imgRef = useRef(null);

  useEffect(() => {
    if (imgRef.current) {
      imgRef.current.src = src;
    }
  }, [src]);

  return (
    <div className="glass-panel overflow-hidden relative flex flex-col h-full w-full">
      <div className="absolute top-0 left-0 w-full p-4 flex justify-between items-center z-10 bg-gradient-to-b from-black/60 to-transparent">
        <h2 className="text-xl font-semibold tracking-wide flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse outline outline-2 outline-emerald-500/50"></span>
          LIVE DYNAMICS
        </h2>
        <div className="text-xs font-mono bg-black/40 px-3 py-1 rounded border border-white/10 uppercase">
          Autonomous SLAM Unit
        </div>
      </div>
      
      <div className="flex-grow flex items-center justify-center bg-black/40">
        {!src ? (
          <div className="text-gray-500 flex flex-col items-center gap-4">
            <svg className="w-12 h-12 animate-spin text-gray-700" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Awaiting Video Input...</span>
          </div>
        ) : (
          <img 
            ref={imgRef}
            alt="AI Annotated Feed"
            className="w-full h-full object-contain"
          />
        )}
      </div>

      <div className="absolute bottom-0 left-0 w-full p-4 z-10 bg-gradient-to-t from-black/80 to-transparent flex justify-between text-xs font-mono text-emerald-400">
        <span>YOLOv11 Perceptive Engine</span>
        <span>A* Path Planning Vector</span>
      </div>
    </div>
  );
}
