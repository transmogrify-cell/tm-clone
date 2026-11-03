import { useEffect } from 'react';
import { AppHeader } from './components/AppHeader';
import { DetectTab } from './components/DetectTab';
import { ExportTab } from './components/ExportTab';
import { Landing } from './components/Landing';
import { PreviewTab } from './components/PreviewTab';
import { TrainTab } from './components/TrainTab';
import { Toaster } from './components/ui/Toaster';
import { useDetectors } from './hooks/useDetectors';
import { useProjectStore } from './store/useProjectStore';

export default function App() {
  useDetectors(); // load COCO-SSD + BlazeFace + Hands (parallel), then MobileNet, right away
  const screen = useProjectStore((s) => s.screen);
  const tab = useProjectStore((s) => s.tab);

  // Samples live only in memory: warn before a reload throws them away.
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      const { classes, trained } = useProjectStore.getState();
      if (classes.some((c) => c.samples.length > 0) || trained) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  return (
    <>
      {screen === 'landing' ? (
        <Landing />
      ) : (
        <div className="min-h-screen">
          <AppHeader />
          {/* Inactive tabs unmount, which stops their frame loops and releases the camera. */}
          {tab === 'detect' && <DetectTab />}
          {tab === 'train' && <TrainTab />}
          {tab === 'preview' && <PreviewTab />}
          {tab === 'export' && <ExportTab />}
        </div>
      )}
      <Toaster />
    </>
  );
}
