import { Build } from './pages/Build'
import { Home } from './pages/Home'
import { Play } from './pages/Play'
import { Watch } from './pages/Watch'
import { navigate, useRoute } from './router'
import { Logo } from './ui/bits'

export default function App() {
  const route = useRoute()
  const [page, ...rest] = route
  const clean = (s?: string) => (s ?? '').split('?')[0]
  let body
  if (page === 'watch') body = <Watch key={route.join('/')} run={clean(rest[0]) || 'sol'} levelId={clean(rest[1]) || 'L00000'} />
  else if (page === 'play') body = <Play key={route.join('/')} levelId={clean(rest[0]) || 'L00000'} />
  else if (page === 'build') body = <Build />
  else body = <Home />

  return (
    <div className="relative h-full w-full overflow-hidden">
      <nav className="absolute inset-x-0 top-0 z-20 flex items-center justify-between px-4 py-3 sm:px-6">
        <button onClick={() => navigate('/')} className="flex items-center gap-2.5">
          <Logo className="h-8" />
          <span className="text-xl font-extrabold tracking-tight">knock</span>
        </button>
        <div className="glass flex items-center gap-1 rounded-full px-1.5 py-1 text-sm">
          <button onClick={() => navigate('/watch/sol/L00000')} className={`rounded-full px-3 py-1.5 ${page === 'watch' ? 'bg-ink text-white' : 'hover:bg-black/5'}`}>Watch</button>
          <button onClick={() => navigate('/play/L00005')} className={`rounded-full px-3 py-1.5 ${page === 'play' ? 'bg-ink text-white' : 'hover:bg-black/5'}`}>Play</button>
          <button onClick={() => navigate('/build')} className={`rounded-full px-3 py-1.5 ${page === 'build' ? 'bg-ink text-white' : 'hover:bg-black/5'}`}>Build</button>
          <a href="https://github.com/srthkdev/knock" className="rounded-full px-3 py-1.5 hover:bg-black/5">GitHub</a>
        </div>
      </nav>
      {body}
    </div>
  )
}
