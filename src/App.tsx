import { useEffect } from 'react';
import { Landing } from './components/Landing';
import { ProjectWorkspace } from './components/ProjectWorkspace';
import { Toaster } from './components/ui/Toaster';
import { useModel } from './hooks/useModel';
import { useProjectStore } from './store/useProjectStore';

export default function App() {
  useModel(); // start loading MobileNet immediately
  const screen = useProjectStore((s) => s.screen);

  // Samples live only in memory: warn before a reload or tab close throws them away.
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
      {screen === 'landing' ? <Landing /> : <ProjectWorkspace />}
      <Toaster />
    </>
  );
}
