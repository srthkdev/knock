import { Build } from './pages/Build'
import { Factory } from './pages/Factory'
import { FactoryWatch } from './pages/FactoryWatch'
import { Home } from './pages/Home'
import { Play } from './pages/Play'
import { Tasks } from './pages/Tasks'
import { Watch } from './pages/Watch'
import { navigate, useRoute } from './router'
import { Logo } from './ui/bits'

// Routes (mode first, then environment and task):
//   #/watch                         task browser (agents)
//   #/watch/<env>/<run>/<task>      watch one agent on one task      (env: cube | factory)
//   #/play                          task browser (you)
//   #/play/<env>/<task>             play one task
//   #/build                         build a cube-lab level from a prompt (local server)
// Legacy cube-lab links (#/watch/<run>/<task>, #/play/<task>) still work.
export default function App() {
  const route = useRoute()
  const [page, ...rest] = route
  const clean = (s?: string) => (s ?? '').split('?')[0]
  const isEnv = (s?: string) => s === 'cube' || s === 'factory'
  let body
  if (page === 'watch') {
    if (!rest.length) body = <Tasks mode="watch" />
    else if (isEnv(rest[0])) {
      const [env, run, task] = rest.map(clean)
      body = env === 'factory'
        ? (run && run !== 'demo' && task ? <FactoryWatch key={route.join('/')} run={run} taskId={task} /> : <Factory key={route.join('/')} />)
        : <Watch key={route.join('/')} run={run || 'sol'} levelId={task || 'L00000'} />
    } else body = <Watch key={route.join('/')} run={clean(rest[0]) || 'sol'} levelId={clean(rest[1]) || 'L00000'} />
  } else if (page === 'play') {
    if (!rest.length) body = <Tasks mode="play" />
    else if (isEnv(rest[0])) body = rest[0] === 'cube' ? <Play key={route.join('/')} levelId={clean(rest[1]) || 'L00005'} /> : <Tasks mode="play" />
    else body = <Play key={route.join('/')} levelId={clean(rest[0]) || 'L00005'} />
  } else if (page === 'factory') body = <Factory key={route.join('/')} />
  else if (page === 'build') body = <Build />
  else body = <Home />

  const tab = (label: string, to: string, on: boolean) => (
    <button onClick={() => navigate(to)} className={`rounded-full px-3 py-1.5 ${on ? 'bg-ink text-white' : 'hover:bg-black/5'}`}>{label}</button>
  )
  return (
    <div className="relative h-full w-full overflow-hidden">
      <nav className="absolute inset-x-0 top-0 z-20 flex items-center justify-between px-4 py-3 sm:px-6">
        <button onClick={() => navigate('/')} className="flex items-center gap-2.5">
          <Logo className="h-8" />
          <span className="text-xl font-extrabold tracking-tight">knock</span>
        </button>
        <div className="glass flex items-center gap-1 rounded-full px-1.5 py-1 text-sm">
          {tab('Watch', '/watch', page === 'watch' || page === 'factory')}
          {tab('Play', '/play', page === 'play')}
          {tab('Build', '/build', page === 'build')}
          <a href="https://github.com/srthkdev/knock" className="rounded-full px-3 py-1.5 hover:bg-black/5">GitHub</a>
        </div>
      </nav>
      {body}
    </div>
  )
}
