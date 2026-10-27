# Changelog entries

Each pull request that changes what people get from Sentient adds one small file here instead of editing
[CHANGELOG.md](../CHANGELOG.md), so pull requests don't collide in that file. At release time the files are
joined into the changelog and deleted.

## Name

`<issue or PR number>.<type>.md`, for example `412.fixed.md`. The type is one of:

| Type | For |
|---|---|
| `added` | something new people can do |
| `changed` | something that works differently now |
| `deprecated` | something that will go away |
| `removed` | something that is gone |
| `fixed` | a bug fix |
| `security` | anything that keeps people or their data safer |

Use the issue number when there is one, or the pull request number. Two entries for the same number:
`412.fixed.md` and `412.fixed.2.md`. No number at all: start the name with `+`, for example `+settings-copy.changed.md`.

## Text

Write for people who use Sentient, not for developers: plain language, a short bold first sentence that says what
changed, then a sentence or two on what it means for them. Leave out the number; it is added at the end for you.

```markdown
**Pausing a task stops it now.** Before, the run already going carried on and could finish while the task showed
Paused.
```

becomes, in CHANGELOG.md under "Fixed":

```markdown
- **Pausing a task stops it now.** Before, the run already going carried on and could finish while the task showed
  Paused (#412).
```

Changes people won't notice (tests, refactoring, build tooling, docs) need no entry. The "Changelog entry" check
asks for one when a pull request changes `sentient/`, `desktop/src/` or `packaging/`; a maintainer can add the
`no changelog` label instead.

## Releasing

```bash
python packaging/release_changelog.py 3.0.0-alpha.1 --draft   # see the new section, change nothing
python packaging/release_changelog.py 3.0.0-alpha.1           # write it to CHANGELOG.md, delete the entries
```
