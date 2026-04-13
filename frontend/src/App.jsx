import React, { useState, useEffect } from 'react';
import VideoFeed from './components/VideoFeed';
import Telemetry from './components/Telemetry';
import { UploadCloud, Camera, Cpu } from 'lucide-react';

function App() {
  const [telemetry, setTelemetry] = useState({});
  const [videoSrc, setVideoSrc] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    // Start WebSocket connection to backend
    const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTelemetry(data);
    };
    
    // Set initial video feed (starts with whatever backend yields)
    setVideoSrc('http://localhost:8000/api/video_feed');

    return () => ws.close();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      await fetch('http://localhost:8000/api/upload_video', {
        method: 'POST',
        body: formData,
      });
      // Force refresh of video source to bypass browser cache
      setVideoSrc(`http://localhost:8000/api/video_feed?t=${new Date().getTime()}`);
    } catch (err) {
      console.error("Error uploading video:", err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-gray-100 flex flex-col font-sans">
      {/* Header */}
      <header className="h-16 glass-panel rounded-none border-t-0 border-x-0 flex items-center justify-between px-6 z-20">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Cpu className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-100 to-gray-400">
              Visual SLAM Protocol
            </h1>
            <p className="text-xs text-indigo-400 font-medium tracking-wider uppercase">Autonomous Navigation Control</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Controls */}
          <label className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg cursor-pointer transition-all active:scale-95">
            {isUploading ? (
               <span className="animate-pulse flex items-center gap-2">
                 <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div> 
                 Processing...
               </span>
            ) : (
              <>
                 <UploadCloud className="w-4 h-4 text-emerald-400" />
                 <span className="text-sm font-medium">Uplink Video</span>
                 <input type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
              </>
            )}
          </label>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow p-6 grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-64px)]">
        
        {/* Left Column: Telemetry */}
        <div className="lg:col-span-1 flex flex-col h-full overflow-hidden">
          <Telemetry data={telemetry} />
        </div>

        {/* Right Columns: Video Feed */}
        <div className="lg:col-span-2 h-full rounded-2xl overflow-hidden shadow-2xl shadow-emerald-500/10">
          <VideoFeed src={videoSrc} />
        </div>

      </main>
      
      {/* Dynamic Aesthetic Background Blobs */}
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-600/10 blur-[120px] pointer-events-none -z-10"></div>
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-emerald-600/10 blur-[120px] pointer-events-none -z-10"></div>
    </div>
  );
}

export default App;
