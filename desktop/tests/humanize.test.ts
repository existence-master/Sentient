// Labels made from config keys and model titles (#454): settings without a hand-written label read as plain words,
// and names like Claude Code keep their spelling.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { humanize } from '../src/lib/utils.ts'

test('a config key reads as sentence case', () => {
  assert.equal(humanize('max_tool_rounds'), 'Max tool rounds')
  assert.equal(humanize('keep-superseded-days'), 'Keep superseded days')
  assert.equal(humanize('suggestionType'), 'Suggestion type')
  assert.equal(humanize(''), '')
})

test('Claude Code keeps its capital C, from the key or from the schema title', () => {
  assert.equal(humanize('claude_code_max_concurrent'), 'Claude Code max concurrent')
  assert.equal(humanize('Claude Code Max Concurrent'), 'Claude Code max concurrent')
  assert.equal(humanize('experimental_claude_code'), 'Experimental Claude Code')
  assert.equal(humanize('claude'), 'Claude')
})

test('product names and short forms keep their spelling', () => {
  assert.equal(humanize('chatgpt_client_id'), 'ChatGPT client ID')
  assert.equal(humanize('github_oauth_client_id'), 'GitHub OAuth client ID')
  assert.equal(humanize('mcp_timeout_s'), 'MCP timeout s')
  assert.equal(humanize('home_assistant_live'), 'Home Assistant live')
  assert.equal(humanize('whatsapp'), 'WhatsApp')
  assert.equal(humanize('tts_first_clause_chars'), 'TTS first clause chars')
  assert.equal(humanize('suggestion_ttl_hours'), 'Suggestion TTL hours')
})

test('mixed-case names survive schema titles and camelCase or PascalCase keys', () => {
  assert.equal(humanize('GitHub OAuth Client ID'), 'GitHub OAuth client ID')
  assert.equal(humanize('ChatGPT Client ID'), 'ChatGPT client ID')
  assert.equal(humanize('OpenAI API Key'), 'OpenAI API key')
  assert.equal(humanize('WhatsApp Bridge'), 'WhatsApp bridge')
  assert.equal(humanize('githubOAuthClientId'), 'GitHub OAuth client ID')
  assert.equal(humanize('myGitHubToken'), 'My GitHub token')
  assert.equal(humanize('GitHubToken'), 'GitHub token')
  assert.equal(humanize('openAIKey'), 'OpenAI key')
  assert.equal(humanize('useChatGPT'), 'Use ChatGPT')
  assert.equal(humanize('whatsAppEnabled'), 'WhatsApp enabled')
  assert.equal(humanize('GITHUB_TOKEN'), 'GitHub token')
})

test('names only match whole words', () => {
  assert.equal(humanize('valid_idea'), 'Valid idea')
  assert.equal(humanize('said_aid'), 'Said aid')
  assert.equal(humanize('claudette'), 'Claudette')
})
