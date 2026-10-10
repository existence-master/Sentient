import { IconCheck, IconDeviceDesktop, IconMoon, IconSun } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { Combobox, FormRow, FormSection, Input, SegmentedControl, Select, Skeleton, Switch, Tooltip } from '@/components/ui'
import { useConfigEditor } from '@/hooks/config'
import { getBridge } from '@/lib/bridge'
import { ACCENT_META, normalizeAccent } from '@/lib/theme'
import { ACCENTS, type ThemePreference } from '@/lib/types'
import { cn, detectTimezone, listTimezones } from '@/lib/utils'
import { useUI } from '@/stores/ui'
import { SchemaForm } from '../SchemaForm'
import type { SectionProps } from '../SettingsPage'
import { ShortcutSettings } from './Shortcuts'

const LANGUAGES = [
  ['en', 'English'],
  ['hi', 'हिन्दी (Hindi)'],
  ['mr', 'मराठी (Marathi)'],
  ['es', 'Español'],
  ['fr', 'Français'],
  ['de', 'Deutsch'],
  ['it', 'Italiano'],
  ['pt', 'Português'],
  ['nl', 'Nederlands'],
  ['ja', '日本語'],
  ['ko', '한국어'],
  ['zh', '中文']
] as const

export function GeneralSection({ query }: SectionProps) {
  const { config, setValue } = useConfigEditor()
  const bridge = getBridge()
  const setTheme = useUI((s) => s.setTheme)
  const setAccent = useUI((s) => s.setAccent)
  const [loginItem, setLoginItem] = useState<boolean | null>(null)

  useEffect(() => {
    if (bridge.isDesktop) void bridge.getLaunchAtLogin().then(setLoginItem)
  }, [bridge])

  if (!config) {
    return (
      <div className="space-y-2">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-12 rounded-xl" />
        ))}
      </div>
    )
  }

  const zones = listTimezones()
  const accent = normalizeAccent(config.ui.accent)
  const launchAtLogin = loginItem ?? config.ui.launch_at_login

  return (
    <div className="space-y-8">
      <FormSection title="Assistant">
        <FormRow label="Assistant name" description="What Sentient calls itself." htmlFor="g-name">
          <Input id="g-name" value={config.assistant.name} onChange={(e) => setValue('assistant.name', e.target.value)} className="w-64" />
        </FormRow>
        <FormRow label="Your name" description="How Sentient addresses you." htmlFor="g-user">
          <Input id="g-user" value={config.assistant.user_name} onChange={(e) => setValue('assistant.user_name', e.target.value)} className="w-64" />
        </FormRow>
        <FormRow label="Timezone" description={`Used for schedules and reminders. This computer is on ${detectTimezone()}.`}>
          <Combobox
            className="w-64"
            value={config.assistant.timezone}
            onChange={(v) => setValue('assistant.timezone', v, { immediate: true })}
            allowCustom={false}
            renderValue={(v) => (v === 'auto' ? `Automatic (${detectTimezone()})` : v.replace(/_/g, ' '))}
            groups={[
              { id: 'auto', label: 'Automatic', options: [{ value: 'auto', label: `Automatic (${detectTimezone()})` }] },
              { id: 'tz', label: 'Timezones', options: zones.map((z) => ({ value: z, label: z.replace(/_/g, ' ') })) }
            ]}
          />
        </FormRow>
        <FormRow label="Location" description="City, Country. For weather, maps and local context." htmlFor="g-loc">
          <Input id="g-loc" value={config.assistant.location} placeholder="Pune, India" onChange={(e) => setValue('assistant.location', e.target.value)} className="w-64" />
        </FormRow>
        <FormRow label="Reply language" description="Sentient answers in this language unless you write in another.">
          <Select
            className="w-64"
            value={LANGUAGES.some(([c]) => c === config.assistant.language) ? config.assistant.language : 'en'}
            onValueChange={(v) => setValue('assistant.language', v, { immediate: true })}
            options={LANGUAGES.map(([value, label]) => ({ value, label }))}
          />
        </FormRow>
      </FormSection>

      <FormSection title="Appearance">
        <FormRow label="Theme">
          <SegmentedControl<ThemePreference>
            value={config.ui.theme}
            onChange={(v) => {
              setTheme(v)
              setValue('ui.theme', v, { immediate: true })
            }}
            options={[
              { value: 'system', label: 'System', icon: <IconDeviceDesktop size={14} /> },
              { value: 'dark', label: 'Dark', icon: <IconMoon size={14} /> },
              { value: 'light', label: 'Light', icon: <IconSun size={14} /> }
            ]}
          />
        </FormRow>
        <FormRow label="Accent color" description="Sentient amber is the brand color.">
          <div role="radiogroup" className="flex items-center gap-2.5">
            {ACCENTS.map((a) => (
              <Tooltip key={a} content={ACCENT_META[a].label}>
                <button
                  type="button"
                  role="radio"
                  aria-checked={accent === a}
                  aria-label={ACCENT_META[a].label}
                  onClick={() => {
                    setAccent(a)
                    setValue('ui.accent', a, { immediate: true })
                  }}
                  className={cn(
                    'flex size-7 items-center justify-center rounded-full ring-offset-2 ring-offset-surface transition-[box-shadow,transform] hover:scale-110',
                    accent === a && 'ring-2 ring-fg/60'
                  )}
                  style={{ background: ACCENT_META[a].color }}
                >
                  {accent === a && <IconCheck size={14} stroke={3} className={a === 'violet' || a === 'rose' ? 'text-white' : 'text-black/80'} />}
                </button>
              </Tooltip>
            ))}
          </div>
        </FormRow>
      </FormSection>

      <SchemaForm section="chat" title="Conversations" filter={query} />

      <FormSection title="Startup & window">
        <FormRow label="Launch at login" description={bridge.isDesktop ? 'Start Sentient quietly in the tray when you sign in.' : 'Available in the desktop app.'}>
          <Switch
            checked={launchAtLogin}
            disabled={!bridge.isDesktop}
            onCheckedChange={(v) => {
              setLoginItem(v)
              void bridge.setLaunchAtLogin(v)
              setValue('ui.launch_at_login', v, { immediate: true })
            }}
          />
        </FormRow>
        <FormRow label="Keep running in the tray" description="Closing the window keeps Sentient working on tasks and suggestions in the background.">
          <Switch
            checked={config.ui.minimize_to_tray}
            onCheckedChange={(v) => {
              void bridge.syncPrefs({ minimizeToTray: v })
              setValue('ui.minimize_to_tray', v, { immediate: true })
            }}
          />
        </FormRow>
      </FormSection>

      <ShortcutSettings />
    </div>
  )
}
