/**
 * Preview data for screens whose engine endpoints are not built yet.
 *
 * Only used when an endpoint answers 404/405/501 AND preview mode is on (`?preview=1` in the
 * hash route, remembered for the window's session). Real installs never see it: the screens
 * show friendly "not available yet" states instead. Screenshots use it before the engine lands.
 */
import { isNotImplemented } from '@/lib/api'
import type { BrowserFrame, BrowserStatus, Channel, DeviceLanInfo, DeviceNode, DevicePairing, Subagent } from '@/lib/types'

const KEY = 'sentient.preview'

export function previewMode(): boolean {
  try {
    if (/[?&]preview=1\b/.test(window.location.hash)) {
      sessionStorage.setItem(KEY, '1')
      return true
    }
    return sessionStorage.getItem(KEY) === '1'
  } catch {
    return false
  }
}

/** Run `load`; when the endpoint doesn't exist yet and preview mode is on, return `fixture()` instead. */
export async function previewOr<T>(load: () => Promise<T>, fixture: () => T): Promise<T> {
  try {
    return await load()
  } catch (err) {
    if (isNotImplemented(err) && previewMode()) return fixture()
    throw err
  }
}

const ago = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString()
const ahead = (minutes: number) => new Date(Date.now() + minutes * 60_000).toISOString()

export const previewNodes = (): DeviceNode[] => [
  {
    node_id: 'desktop',
    name: 'This computer',
    kind: 'desktop',
    platform: 'win32',
    capabilities: ['screen.capture', 'camera.photo', 'clipboard.read', 'clipboard.write', 'notify.show'],
    online: true,
    last_seen_at: ago(0),
    battery: null,
    created_at: ago(60 * 24 * 12)
  },
  {
    node_id: 'node-phone',
    name: "Maya's phone",
    kind: 'phone',
    platform: 'android',
    capabilities: ['camera.photo', 'location.get', 'notify.show', 'speak', 'mic.stream', 'battery'],
    online: true,
    last_seen_at: ago(1),
    battery: 0.72,
    charging: false,
    created_at: ago(60 * 24 * 4)
  },
  {
    node_id: 'node-glasses',
    name: 'Smart glasses',
    kind: 'glasses',
    platform: 'glasses',
    capabilities: ['camera.photo', 'display.text', 'display.card', 'mic.stream', 'button.events', 'battery', 'speak'],
    online: false,
    last_seen_at: ago(185),
    battery: 0.34,
    created_at: ago(60 * 24 * 2)
  }
]

export const previewLan = (): DeviceLanInfo => ({
  enabled: false,
  port: 7443,
  urls: ['https://192.168.1.24:7443'],
  fingerprint: 'SHA256 4F:9A:21:C7:0B:6E:D3:88:15:AF:42:90:7C:E1:5B:D6'
})

export const previewPairing = (): DevicePairing => ({
  code: '482913',
  expires_at: ahead(10),
  lan_enabled: false,
  urls: ['https://192.168.1.24:7443'],
  qr: 'sentient://pair?url=https%3A%2F%2F192.168.1.24%3A7443&code=482913'
})

const TELEGRAM_SETUP = `Telegram bots are free and take about a minute to make.

1. Open Telegram and start a chat with **@BotFather** (https://t.me/BotFather).
2. Send \`/newbot\`, then pick a name and a username ending in "bot".
3. BotFather replies with a token that looks like \`123456:ABC-DEF...\`. Copy it and paste it below.

Only chats you pair with a code can talk to Sentient through your bot.`

const DISCORD_SETUP = `1. Open the Discord Developer Portal (https://discord.com/developers/applications) and choose **New Application**.
2. Open **Bot**, choose **Reset Token** and copy the token.
3. Under **Privileged Gateway Intents**, turn on **Message Content Intent**.
4. Open **OAuth2**, tick \`bot\`, and use the link to add the bot to your server.

Only chats you pair with a code can talk to Sentient through your bot.`

export const previewChannels = (): Channel[] => [
  {
    id: 'telegram',
    display_name: 'Telegram',
    status: 'connected',
    account_label: '@maya_sentient_bot',
    error: null,
    paired: [{ chat_id: '100200301', label: 'Maya Rao', paired_at: ago(60 * 50), deliver: true, session_id: 'demo-chat-telegram' }],
    setup: { fields: [{ key: 'bot_token', label: 'Bot token', secret: true, required: true }], instructions_md: TELEGRAM_SETUP }
  },
  {
    id: 'discord',
    display_name: 'Discord',
    status: 'disconnected',
    account_label: null,
    error: null,
    paired: [],
    setup: { fields: [{ key: 'bot_token', label: 'Bot token', secret: true, required: true }], instructions_md: DISCORD_SETUP }
  }
]

export const previewSubagents = (sessionId: string): Subagent[] =>
  sessionId === 'demo-chat-background' || sessionId === 'demo-chat-main'
    ? [
        {
          subagent_id: 'sub-demo-2',
          session_id: sessionId,
          parent_call_id: null,
          goal: 'Compare 4 standing desks under 40,000 rupees available in Bengaluru',
          status: 'running',
          background: true,
          summary: null,
          error: null,
          tool_calls: 3,
          started_at: ago(4),
          finished_at: null
        }
      ]
    : []

export const previewSubagent = (id: string): Subagent => ({
  subagent_id: id,
  session_id: 'demo-chat-main',
  parent_call_id: null,
  goal: 'Shortlist 3 quiet restaurants in Indiranagar',
  status: 'completed',
  background: false,
  summary: null,
  error: null,
  tool_calls: 6,
  started_at: ago(12),
  finished_at: ago(10)
})

const PAGE_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" viewBox="0 0 960 600">
<rect width="960" height="600" fill="#fbfaf7"/>
<rect width="960" height="64" fill="#2f6f6a"/>
<text x="32" y="41" font-family="Segoe UI, Arial" font-size="24" font-weight="700" fill="#fff">tablefinder</text>
<rect x="180" y="18" width="520" height="30" rx="15" fill="#fff" opacity=".95"/>
<text x="200" y="38" font-family="Segoe UI, Arial" font-size="13" fill="#888">Indiranagar, Bengaluru</text>
<rect x="32" y="96" width="420" height="260" rx="14" fill="#d9c7a8"/>
<circle cx="150" cy="210" r="70" fill="#c4a57a"/><circle cx="310" cy="190" r="54" fill="#8f6f45"/>
<rect x="484" y="96" width="444" height="36" rx="6" fill="none"/>
<text x="484" y="124" font-family="Segoe UI, Arial" font-size="30" font-weight="700" fill="#1c1c1c">The Flour Works</text>
<text x="484" y="156" font-family="Segoe UI, Arial" font-size="15" fill="#696969">European, Cafe, Vegetarian friendly</text>
<text x="484" y="182" font-family="Segoe UI, Arial" font-size="15" fill="#696969">12th Main, Indiranagar, 6 min walk from the metro</text>
<rect x="484" y="206" width="64" height="28" rx="6" fill="#267e3e"/>
<text x="498" y="226" font-family="Segoe UI, Arial" font-size="15" font-weight="700" fill="#fff">4.4</text>
<rect x="484" y="262" width="200" height="48" rx="10" fill="#2f6f6a"/>
<text x="524" y="292" font-family="Segoe UI, Arial" font-size="17" font-weight="600" fill="#fff">Book a table</text>
<rect x="700" y="262" width="150" height="48" rx="10" fill="#fff" stroke="#2f6f6a"/>
<text x="740" y="292" font-family="Segoe UI, Arial" font-size="17" fill="#2f6f6a">Menu</text>
<rect x="32" y="388" width="896" height="1" fill="#e8e8e8"/>
<text x="32" y="428" font-family="Segoe UI, Arial" font-size="18" font-weight="600" fill="#1c1c1c">Saturday, 8:00 PM</text>
<rect x="32" y="448" width="120" height="40" rx="8" fill="#fff" stroke="#ddd"/><text x="62" y="474" font-family="Segoe UI, Arial" font-size="15" fill="#333">7:30 PM</text>
<rect x="164" y="448" width="120" height="40" rx="8" fill="#e6f2f1" stroke="#2f6f6a"/><text x="194" y="474" font-family="Segoe UI, Arial" font-size="15" fill="#2f6f6a">8:00 PM</text>
<rect x="296" y="448" width="120" height="40" rx="8" fill="#fff" stroke="#ddd"/><text x="326" y="474" font-family="Segoe UI, Arial" font-size="15" fill="#333">8:30 PM</text>
<text x="32" y="530" font-family="Segoe UI, Arial" font-size="15" fill="#696969">2 guests</text>
</svg>`

export const previewFrameImage = `data:image/svg+xml;base64,${btoa(PAGE_SVG)}`

export const previewBrowser = (): BrowserStatus => ({
  available: true,
  running: true,
  engine: 'Microsoft Edge',
  headless: true,
  tabs: [
    { index: 0, url: 'https://tablefinder.example/bengaluru/the-flour-works/book', title: 'Book a table - The Flour Works', active: true },
    { index: 1, url: 'https://maps.example/place/indiranagar-metro', title: 'Indiranagar metro - Maps', active: false }
  ],
  error: null
})

export const previewFrame = (): BrowserFrame => ({
  url: 'https://tablefinder.example/bengaluru/the-flour-works/book',
  title: 'Book a table - The Flour Works',
  image: previewFrameImage
})

const PHOTO_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2f5d3a"/><stop offset="1" stop-color="#a8c686"/></linearGradient></defs><rect width="640" height="480" fill="url(#g)"/><ellipse cx="320" cy="260" rx="160" ry="120" fill="#3f7a45" opacity=".8"/><ellipse cx="250" cy="220" rx="70" ry="40" fill="#6fae5b"/><ellipse cx="390" cy="230" rx="80" ry="44" fill="#5c9a4e"/><rect x="270" y="360" width="100" height="90" rx="10" fill="#b5653b"/></svg>`

export const previewPhotoBase64 = btoa(PHOTO_SVG)
