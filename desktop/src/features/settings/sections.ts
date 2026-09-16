import {
  IconBolt,
  IconBrain,
  IconChartBar,
  IconCode,
  IconCpu,
  IconListCheck,
  IconMicrophone,
  IconMoodSmile,
  IconSettings,
  IconShieldCheck,
  IconSparkles,
  IconTool,
  IconUserHeart,
  IconUsersGroup,
  IconWorldWww,
  type Icon
} from '@tabler/icons-react'

export interface SettingsSectionMeta {
  id: string
  label: string
  description: string
  icon: Icon
  /** Extra words matched by the settings search and the command palette. */
  keywords: string[]
  /** Config sections rendered by the schema-driven form (if any). */
  schemaSections?: string[]
}

export const SETTINGS_SECTIONS: SettingsSectionMeta[] = [
  {
    id: 'general',
    label: 'General',
    description: 'Names, timezone, language, appearance and startup.',
    icon: IconSettings,
    keywords: ['name', 'timezone', 'location', 'language', 'theme', 'dark', 'light', 'accent', 'color', 'login', 'startup', 'tray']
  },
  {
    id: 'models',
    label: 'Models',
    description: 'Which model does which job, API keys and local models.',
    icon: IconCpu,
    keywords: ['llm', 'ollama', 'openai', 'anthropic', 'api key', 'provider', 'embedding', 'fallback', 'reasoning', 'temperature', 'pull']
  },
  {
    id: 'personality',
    label: 'Personality',
    description: "Sentient's soul and what it knows about you.",
    icon: IconMoodSmile,
    keywords: ['soul', 'persona', 'user.md', 'tone', 'voice', 'about me']
  },
  {
    id: 'memory',
    label: 'Memory',
    description: 'How Sentient learns and recalls facts about you.',
    icon: IconBrain,
    keywords: ['facts', 'recall', 'similarity', 'summaries', 'workspace'],
    schemaSections: ['memory']
  },
  {
    id: 'knowing',
    label: 'Knowing you',
    description: 'How Sentient builds its picture of you and tidies its memory overnight.',
    icon: IconUserHeart,
    keywords: ['user model', 'insights', 'about you', 'dreams', 'dreaming', 'consolidation', 'overnight', 'questions'],
    schemaSections: ['user_model', 'dreaming']
  },
  {
    id: 'tasks',
    label: 'Tasks',
    description: 'Long-running work, scheduling and plan approval.',
    icon: IconListCheck,
    keywords: ['scheduler', 'runs', 'swarm', 'plan', 'timeout'],
    schemaSections: ['tasks']
  },
  {
    id: 'proactivity',
    label: 'Proactivity',
    description: 'Suggestions from your connected apps.',
    icon: IconBolt,
    keywords: ['suggestions', 'quiet hours', 'poll', 'confidence', 'heartbeat'],
    schemaSections: ['proactivity']
  },
  {
    id: 'voice',
    label: 'Voice',
    description: 'Speech recognition and the voice Sentient speaks with.',
    icon: IconMicrophone,
    keywords: ['speech', 'stt', 'tts', 'whisper', 'dictation', 'wake word', 'hey sentient', 'always listening', 'follow-up']
  },
  {
    id: 'approvals',
    label: 'Approvals & safety',
    description: 'When Sentient asks before acting.',
    icon: IconShieldCheck,
    keywords: ['permissions', 'confirm', 'risk', 'tools', 'disabled', 'ask'],
    schemaSections: ['tools']
  },
  {
    id: 'sandbox',
    label: 'Code execution',
    description: 'Small scripts Sentient writes and runs to get things done.',
    icon: IconCode,
    keywords: ['sandbox', 'python', 'docker', 'scripts', 'execute', 'code', 'watchers'],
    schemaSections: ['sandbox']
  },
  {
    id: 'browser',
    label: 'Browser',
    description: 'The web browser Sentient uses for sites without an integration.',
    icon: IconWorldWww,
    keywords: ['browser', 'chrome', 'edge', 'websites', 'shopping', 'purchases', 'live view'],
    schemaSections: ['browser']
  },
  {
    id: 'subagents',
    label: 'Helpers',
    description: 'Helpers Sentient can hand parts of a bigger job to.',
    icon: IconUsersGroup,
    keywords: ['subagents', 'delegate', 'parallel', 'background', 'workers', 'helpers'],
    schemaSections: ['subagents']
  },
  {
    id: 'evolution',
    label: 'Self-evolution',
    description: 'Skills Sentient writes for itself and how it keeps them tidy.',
    icon: IconSparkles,
    keywords: ['skills', 'curator', 'review', 'profile'],
    schemaSections: ['evolution', 'skills']
  },
  {
    id: 'usage',
    label: 'Usage',
    description: 'Tokens by model, source and day.',
    icon: IconChartBar,
    keywords: ['tokens', 'cost', 'insights', 'statistics']
  },
  {
    id: 'advanced',
    label: 'Advanced',
    description: 'Data folder, logs, engine and raw configuration.',
    icon: IconTool,
    keywords: ['logs', 'folder', 'restart', 'version', 'json', 'debug', 'gateway', 'integrations']
  }
]

export const sectionById = (id: string | undefined) => SETTINGS_SECTIONS.find((s) => s.id === id)
