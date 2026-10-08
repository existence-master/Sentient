# Desktop guide (`desktop/`)

Read the root [AGENTS.md](../AGENTS.md) first.

## Shape

- `electron/main/`: the main process. It picks a free port and a token, starts the engine (`backend.ts`, using the
  frozen engine in installed builds and `.venv` in development), owns the window, tray, notifications and the
  desktop's own device connection (`node.ts`).
- `electron/preload/`: the small, typed bridge exposed as `window.sentient`. Keep it minimal; context isolation and
  sandboxing stay on.
- `src/`: the React renderer (React 19, TypeScript, Tailwind 4, Radix, zustand, React Query).
  - `lib/api.ts`, `lib/types.ts`, `lib/events.ts`: the typed REST client, shared types and domain-event wiring.
    They mirror [docs/API.md](../docs/API.md); change them together.
  - `features/<area>/`: screens (`<Area>Page.tsx`) and components per area. `components/ui/`: the design system;
    reuse it.
  - `hooks/`: React Query hooks per area, with every query key in `hooks/queryKeys.ts`.
  - `lib/demo.ts`: dev-only demo data behind `#/<route>?demo=1`, never active in a packaged build.
  - `stores/`: zustand stores for app-wide state (chat stream, connection, UI).

## Conventions

- Server data goes through React Query; live updates come from domain events that invalidate or patch queries.
- Copy is plain, warm and short. Users are not engineers. No em-dashes.
- Every screen handles loading, empty and error states, and never crashes when an endpoint is missing.
- Demo and seed data use the fictional persona from `scripts/seed-*.py`. Never real people, companies or numbers.

## Checking your work

```bash
npm run typecheck          # must be clean
npm run build              # production build into out/
node scripts/smoke.mjs chat shot.png 1440x900   # launch the built app, capture a route, quit
```

Point `SENTIENT_HOME` at an empty folder and run the seed scripts with the engine's Python in this order, then
capture with the same `SENTIENT_HOME`: `seed-memory-skills.py --reset`, `seed-chats-devices-channels.py --add`,
`seed-tasks.py`, `seed-integrations-notifications.py --keep-db`, `seed-automations-usermodel.py`. They all use one fictional persona (Maya Rao). In Git Bash write routes without a leading slash. When several people or agents build at once,
take `desktop/.build.lock` before `npm run build` and remove it after, because `out/` is shared.
