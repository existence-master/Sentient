/**
 * Pure helpers for "Ask about selected text" (#173). No imports, so `npm test` runs them in Node.
 *
 * The shell copies the selection with the app's own copy keys, reads the clipboard and puts the old clipboard back.
 * Windows uses one small PowerShell helper that stays open for a while after a use (starting PowerShell takes
 * seconds); macOS uses osascript and Linux xdotool, one command per use.
 */

/** The longest selection sent on; longer text is cut and the window says so. */
export const MAX_SELECTION_CHARS = 20_000
/** How long the shell waits for the copied text to reach the clipboard. */
export const COPY_WAIT_MS = 1200
/** The Windows helper closes after this long without a use. */
export const HELPER_IDLE_MS = 30 * 60_000

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/** Where the ask window goes: just below and right of the pointer, kept on the screen, above it when there's no room. */
export function askBounds(cursor: { x: number; y: number }, workArea: Rect, width: number, height: number, gap = 14): Rect {
  const w = Math.min(width, workArea.width)
  const h = Math.min(height, workArea.height)
  const right = workArea.x + workArea.width
  const bottom = workArea.y + workArea.height
  let x = cursor.x + gap
  if (x + w > right) x = cursor.x - gap - w
  x = Math.max(workArea.x, Math.min(x, right - w))
  let y = cursor.y + gap
  if (y + h > bottom) y = cursor.y - gap - h
  y = Math.max(workArea.y, Math.min(y, bottom - h))
  return { x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h) }
}

/** The window keeps its top edge and grows or shrinks to fit what it shows, staying on the screen. */
export function resizedBounds(current: Rect, workArea: Rect, height: number, min = 180): Rect {
  const h = Math.round(Math.max(min, Math.min(height, workArea.height)))
  const bottom = workArea.y + workArea.height
  const y = Math.max(workArea.y, Math.min(current.y, bottom - h))
  return { x: current.x, y, width: current.width, height: h }
}

/** The selected text as it is sent on: line ends made plain, cut at `max` characters. Blank means nothing. */
export function cleanSelection(text: string, max = MAX_SELECTION_CHARS): { text: string; truncated: boolean } {
  const plain = String(text ?? '').replace(/\r\n?/g, '\n')
  if (!plain.trim()) return { text: '', truncated: false }
  if (plain.length <= max) return { text: plain, truncated: false }
  return { text: plain.slice(0, max), truncated: true }
}

/** The app's name as people know it: "Notepad", "Google Chrome". Falls back to the process name, then "another app". */
export function appName(description: string | undefined, processName: string | undefined): string {
  const pick = (s: string | undefined) => String(s ?? '').replace(/\.exe$/i, '').replace(/\s+/g, ' ').trim()
  const name = pick(description) || pick(processName)
  if (!name) return 'another app'
  return name.length > 40 ? name.slice(0, 40).trim() : name.charAt(0).toUpperCase() + name.slice(1)
}

// ------------------------------------------------------------------ Windows helper
/** The Windows key code of an accelerator's main key (`CommandOrControl+Alt+Shift+A` -> 65), 0 when unknown. */
export function acceleratorKeyCode(accelerator: string): number {
  const key = String(accelerator ?? '').split('+').pop()?.trim() ?? ''
  if (/^[a-z0-9]$/i.test(key)) return key.toUpperCase().charCodeAt(0)
  const f = key.match(/^F(\d{1,2})$/i)
  if (f && Number(f[1]) >= 1 && Number(f[1]) <= 24) return 0x6f + Number(f[1])
  return ({ space: 0x20, enter: 0x0d, return: 0x0d, tab: 0x09 } as Record<string, number>)[key.toLowerCase()] ?? 0
}

/** What the Windows helper reports for `capture`. */
export interface CaptureReport {
  /** `copied`: the copy keys were pressed and the clipboard changed; `nothing`: it didn't (no selection);
   *  `password`: a password box had focus, so nothing was copied; `own`: a Sentient window was in front, so nothing
   *  was pressed; `failed`: something went wrong. */
  result: 'copied' | 'nothing' | 'password' | 'own' | 'failed'
  /** The window the text came from, to paste back into (a number as text). */
  window: string
  app: string
  pid: number
  error?: string
}

/** One line of JSON from a capture command, read tolerantly. */
export function parseCapture(line: string): CaptureReport {
  let raw: Record<string, unknown> = {}
  try {
    const found = String(line ?? '').match(/\{.*\}/s)
    raw = found ? (JSON.parse(found[0]) as Record<string, unknown>) : {}
  } catch {
    raw = {}
  }
  const result = ['copied', 'nothing', 'password', 'own'].includes(String(raw.result)) ? (raw.result as CaptureReport['result']) : 'failed'
  return {
    result,
    window: /^\d+$/.test(String(raw.window ?? '')) ? String(raw.window) : '',
    app: appName(raw.description as string | undefined, raw.process as string | undefined),
    pid: Number.isInteger(raw.pid) ? (raw.pid as number) : 0,
    ...(typeof raw.error === 'string' ? { error: raw.error } : {})
  }
}

/** What `forget` reported: how many clipboard history entries it took out, `off` when Windows keeps no clipboard
 *  history, `failed` when it couldn't look. */
export function parseForget(line: string): { result: 'forgot'; count: number } | { result: 'off' | 'failed' } {
  const words = String(line ?? '').trim().toLowerCase().split(/\s+/)
  if (words[0] === 'off') return { result: 'off' }
  if (words[0] === 'forgot' && /^\d+$/.test(words[1] ?? '')) return { result: 'forgot', count: Number(words[1]) }
  return { result: 'failed' }
}

/** The `forget` command for the Windows helper: entries holding `text` added since `since` (ms) leave clipboard history. */
export function forgetCommand(text: string, since: number): string {
  return `forget ${Math.max(0, Math.floor(since))} ${Buffer.from(String(text ?? ''), 'utf8').toString('base64')}`
}

/** What a paste-back command reported. */
export function parsePaste(line: string): 'pasted' | 'password' | 'failed' {
  const word = String(line ?? '').trim().toLowerCase().split(/\s+/).pop() ?? ''
  return word === 'pasted' || word === 'password' ? word : 'failed'
}

/**
 * The Windows helper: PowerShell reading one command per line on stdin and answering one line on stdout.
 *   capture <vk> <pid> -> {"result", "window", "pid", "process", "description"} (vk: the shortcut's key, 0 for none;
 *                         pid: Sentient's own process, whose windows are never copied from)
 *   paste <hwnd>  -> pasted | password | failed
 *   forget <since ms> <base64 utf-8 text> -> forgot <n> | off | failed (#332: the copy Ctrl+C made, or the answer
 *                         Sentient pasted, leaves Windows clipboard history (Win+V); only entries added since `since`
 *                         with exactly that text, so the user's own older copies stay)
 * It waits for the shortcut's keys to be let go before pressing Ctrl+C or Ctrl+V, so the app doesn't see
 * Ctrl+Alt+Shift+C, and never copies from or pastes into a password box.
 */
export const WIN_HELPER = String.raw`$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
public static class SentientAsk {
  [DllImport("user32.dll")] static extern short GetAsyncKeyState(int key);
  [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] static extern bool BringWindowToTop(IntPtr h);
  [DllImport("user32.dll")] static extern bool IsWindow(IntPtr h);
  [DllImport("user32.dll")] static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] static extern bool AttachThreadInput(uint a, uint b, bool attach);
  [DllImport("kernel32.dll")] static extern uint GetCurrentThreadId();
  [DllImport("user32.dll")] static extern uint GetClipboardSequenceNumber();
  [DllImport("user32.dll")] static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
  static readonly int[] Held = { 0x10, 0x11, 0x12, 0x5B, 0x5C };
  public static bool KeysUp(int ms, int key) {
    var sw = Stopwatch.StartNew();
    while (sw.ElapsedMilliseconds < ms) {
      bool down = key > 0 && (GetAsyncKeyState(key) & 0x8000) != 0;
      foreach (int k in Held) if ((GetAsyncKeyState(k) & 0x8000) != 0) { down = true; break; }
      if (!down) return true;
      Thread.Sleep(15);
    }
    return false;
  }
  static void CtrlChord(byte key) {
    keybd_event(0x11, 0, 0, UIntPtr.Zero);
    keybd_event(key, 0, 0, UIntPtr.Zero);
    keybd_event(key, 0, 2, UIntPtr.Zero);
    keybd_event(0x11, 0, 2, UIntPtr.Zero);
  }
  public static long Foreground() { return GetForegroundWindow().ToInt64(); }
  public static uint Pid(long hwnd) { uint pid; GetWindowThreadProcessId(new IntPtr(hwnd), out pid); return pid; }
  public static bool Copy(int ms) {
    uint before = GetClipboardSequenceNumber();
    CtrlChord(0x43);
    var sw = Stopwatch.StartNew();
    while (sw.ElapsedMilliseconds < ms) {
      if (GetClipboardSequenceNumber() != before) return true;
      Thread.Sleep(10);
    }
    return false;
  }
  public static bool Focus(long hwnd, int ms) {
    IntPtr h = new IntPtr(hwnd);
    if (!IsWindow(h)) return false;
    if (IsIconic(h)) ShowWindow(h, 9);
    if (GetForegroundWindow() != h) {
      uint ignored;
      uint fg = GetWindowThreadProcessId(GetForegroundWindow(), out ignored);
      uint me = GetCurrentThreadId();
      bool attached = fg != 0 && fg != me && AttachThreadInput(me, fg, true);
      SetForegroundWindow(h);
      BringWindowToTop(h);
      if (attached) AttachThreadInput(me, fg, false);
    }
    var sw = Stopwatch.StartNew();
    while (sw.ElapsedMilliseconds < ms) {
      if (GetForegroundWindow() == h) return true;
      Thread.Sleep(15);
    }
    return false;
  }
  public static void Paste() { CtrlChord(0x56); }
}
'@
$uia = $true
try { Add-Type -AssemblyName UIAutomationClient } catch { $uia = $false }
function Test-Password {
  if (-not $uia) { return $false }
  try { $f = [System.Windows.Automation.AutomationElement]::FocusedElement; return [bool]($f -and $f.Current.IsPassword) } catch { return $false }
}
$winrt = $null
$asTask = $null
function Initialize-History {
  if ($null -ne $script:winrt) { return $script:winrt }
  try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $script:asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -match '^IAsyncOperation.1$' } | Select-Object -First 1
    $null = [Windows.ApplicationModel.DataTransfer.Clipboard, Windows.ApplicationModel.DataTransfer, ContentType = WindowsRuntime]
    $null = [Windows.ApplicationModel.DataTransfer.ClipboardHistoryItemsResult, Windows.ApplicationModel.DataTransfer, ContentType = WindowsRuntime]
    $script:winrt = $null -ne $script:asTask
  } catch { $script:winrt = $false }
  return $script:winrt
}
function Wait-WinRT($op, [type]$type) {
  $task = $script:asTask.MakeGenericMethod($type).Invoke($null, @($op))
  if (-not $task.Wait(3000)) { throw 'timed out' }
  return $task.Result
}
function Remove-FromHistory([long]$since, [string]$encoded) {
  if (-not (Initialize-History)) { return 'failed' }
  $nl = [string][char]10
  $crlf = [string][char]13 + $nl
  $want = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encoded)).Replace($crlf, $nl)
  $clip = [Windows.ApplicationModel.DataTransfer.Clipboard]
  if (-not $clip::IsHistoryEnabled()) { return 'off' }
  $from = [DateTimeOffset]::FromUnixTimeMilliseconds($since)
  $text = [Windows.ApplicationModel.DataTransfer.StandardDataFormats]::Text
  $n = 0
  $sw = [Diagnostics.Stopwatch]::StartNew()
  # Windows adds a copy to its history a moment after it lands on the clipboard, so look again for a little while
  do {
    $res = Wait-WinRT ($clip::GetHistoryItemsAsync()) ([Windows.ApplicationModel.DataTransfer.ClipboardHistoryItemsResult])
    if ([string]$res.Status -ne 'Success') { return 'off' }
    foreach ($item in @($res.Items)) {
      try {
        if ($item.Timestamp -lt $from -or -not $item.Content.Contains($text)) { continue }
        $got = Wait-WinRT ($item.Content.GetTextAsync()) ([string])
        if ($got -and $got.Replace($crlf, $nl) -ceq $want -and $clip::DeleteItemFromHistory($item)) { $n++ }
      } catch {}
    }
    if ($n -gt 0) { break }
    Start-Sleep -Milliseconds 150
  } while ($sw.ElapsedMilliseconds -lt 2500)
  return "forgot $n"
}
[Console]::Out.WriteLine('ready')
[Console]::Out.Flush()
while ($null -ne ($line = [Console]::In.ReadLine())) {
  $parts = $line.Trim().Split(' ')
  $out = ''
  try {
    if ($parts[0] -eq 'capture') {
      $key = 0
      if ($parts.Length -gt 1) { $key = [int]$parts[1] }
      $own = 0
      if ($parts.Length -gt 2) { $own = [int]$parts[2] }
      $up = [SentientAsk]::KeysUp(1500, $key)
      $hwnd = [SentientAsk]::Foreground()
      $procId = [SentientAsk]::Pid($hwnd)
      $process = ''; $description = ''
      try { $p = Get-Process -Id $procId; $process = $p.ProcessName; $description = $p.MainModule.FileVersionInfo.FileDescription } catch {}
      if (-not $up) { $result = 'failed' }
      elseif ($own -gt 0 -and $procId -eq $own) { $result = 'own' }
      elseif (Test-Password) { $result = 'password' }
      elseif ([SentientAsk]::Copy(1000)) { $result = 'copied' }
      else { $result = 'nothing' }
      $out = @{ result = $result; window = [string]$hwnd; pid = [int]$procId; process = $process; description = $description } | ConvertTo-Json -Compress
    } elseif ($parts[0] -eq 'paste' -and $parts.Length -gt 1) {
      if (-not [SentientAsk]::Focus([long]$parts[1], 1500)) { $out = 'failed' }
      else {
        [void][SentientAsk]::KeysUp(1000, 0)
        Start-Sleep -Milliseconds 60
        if (Test-Password) { $out = 'password' } else { [SentientAsk]::Paste(); $out = 'pasted' }
      }
    } elseif ($parts[0] -eq 'forget' -and $parts.Length -gt 2) {
      $out = Remove-FromHistory ([long]$parts[1]) $parts[2]
    } else { $out = 'failed' }
  } catch {
    $out = @{ result = 'failed'; error = $_.Exception.Message } | ConvertTo-Json -Compress
  }
  [Console]::Out.WriteLine($out)
  [Console]::Out.Flush()
}`

// ------------------------------------------------------------------ macOS and Linux
export interface Command {
  file: string
  args: string[]
}

const MAC_CAPTURE = [
  'use framework "AppKit"',
  'use scripting additions',
  // wait (up to 1.5 s) until Shift, Control, Option and Command are let go, so the app sees Command+C only
  'repeat 50 times',
  "if ((current application's NSEvent's modifierFlags()) div 131072) mod 16 = 0 then exit repeat",
  'delay 0.03',
  'end repeat',
  'tell application "System Events"',
  'set p to first application process whose frontmost is true',
  'set n to name of p',
  'set i to unix id of p',
  'try',
  'set f to value of attribute "AXFocusedUIElement" of p',
  'if (value of attribute "AXSubrole" of f) is "AXSecureTextField" then return "{\\"result\\":\\"password\\",\\"pid\\":" & i & ",\\"process\\":\\"" & n & "\\"}"',
  'end try',
  'keystroke "c" using command down',
  'end tell',
  'return "{\\"result\\":\\"copied\\",\\"window\\":\\"" & i & "\\",\\"pid\\":" & i & ",\\"process\\":\\"" & n & "\\"}"'
]

/** The command that presses the copy keys in the app in front and says which app it was. */
export function captureCommand(platform: string): Command | null {
  if (platform === 'darwin') return { file: 'osascript', args: MAC_CAPTURE.flatMap((line) => ['-e', line]) }
  if (platform === 'linux') {
    const script = [
      'sleep 0.3',
      'w=$(xdotool getactivewindow) || exit 1',
      'n=$(xdotool getwindowname "$w" | tr -d \'"\\\\\' | head -c 60)',
      'xdotool key --clearmodifiers ctrl+c',
      'printf \'{"result":"copied","window":"%s","process":"%s"}\\n\' "$w" "$n"'
    ].join('\n')
    return { file: 'sh', args: ['-c', script] }
  }
  return null
}

/** The command that brings `target` (from a capture) back to the front and pastes there. */
export function pasteBackCommand(platform: string, target: string): Command | null {
  if (!/^\d+$/.test(target)) return null
  if (platform === 'darwin') {
    return {
      file: 'osascript',
      args: [
        'tell application "System Events"',
        `set frontmost of (first application process whose unix id is ${target}) to true`,
        'delay 0.3',
        'try',
        'set f to value of attribute "AXFocusedUIElement" of (first application process whose frontmost is true)',
        'if (value of attribute "AXSubrole" of f) is "AXSecureTextField" then return "password"',
        'end try',
        'keystroke "v" using command down',
        'end tell',
        'return "pasted"'
      ].flatMap((line) => ['-e', line])
    }
  }
  if (platform === 'linux') {
    return { file: 'sh', args: ['-c', `xdotool windowactivate --sync ${target} key --clearmodifiers ctrl+v && echo pasted`] }
  }
  return null
}
