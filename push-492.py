#!/usr/bin/env python3
"""Push local HEAD to azmi2409/Sentient branch fix-492-false-claims via the GitHub git-database API."""
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response

CRED = "custom.github"
ALLOWED = ["api.github.com"]
BASE = "https://api.github.com"
UP = "existence-master/Sentient"
FORK = "azmi2409/Sentient"
BRANCH = "fix-492-false-claims"
WORKDIR = "/home/hatch/workspace/contrib/sentient-492"


def req(path, method="GET", data=None):
    url = BASE + path
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "muse-agent"}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    add_surrogate_to_request(r, CRED, allowed_hosts=ALLOWED)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, read_json_response(resp)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def git(*args):
    return subprocess.run(["git", *args], cwd=WORKDIR, capture_output=True, text=True, check=True).stdout


# files changed in the local HEAD commit (exact case from git)
files = git("show", "--name-only", "--pretty=format:", "HEAD").split()
print("files:", files)

st, ref = req(f"/repos/{UP}/git/refs/heads/main")
assert st == 200, (st, ref)
base_sha = ref["object"]["sha"]
st, base_commit = req(f"/repos/{UP}/git/commits/{base_sha}")
assert st == 200, (st, base_commit)
base_tree = base_commit["tree"]["sha"]
print("upstream main:", base_sha[:7], "tree:", base_tree[:7])

entries = []
for rel in files:
    full = git("show", f"HEAD:{rel}").encode()
    st, blob = req(f"/repos/{FORK}/git/blobs", "POST",
                   {"content": base64.b64encode(full).decode(), "encoding": "base64"})
    assert st == 201, (st, blob, rel)
    entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
print("blobs done:", len(entries))

st, tree = req(f"/repos/{FORK}/git/trees", "POST", {"base_tree": base_tree, "tree": entries})
assert st == 201, (st, tree)
print("tree:", tree["sha"])

st, commit = req(f"/repos/{FORK}/git/commits", "POST", {
    "message": git("log", "-1", "--pretty=%B", "HEAD").strip(),
    "tree": tree["sha"],
    "parents": [base_sha],
})
assert st == 201, (st, commit)
print("commit:", commit["sha"])

st, result = req(f"/repos/{FORK}/git/refs", "POST",
                 {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})
if st == 422:
    print("branch exists, updating")
    st, result = req(f"/repos/{FORK}/git/refs/heads/{BRANCH}", "PATCH", {"sha": commit["sha"], "force": True})
assert st in (200, 201), (st, result)
print("ref ok:", st)

# verify: pushed tree must be byte-identical to the local HEAD tree
st, pushed = req(f"/repos/{FORK}/git/trees/{commit['tree']['sha']}?recursive=1")
assert st == 200, (st, pushed)
local = {}
for line in git("ls-tree", "-r", "HEAD").splitlines():
    # "100644 blob <sha>\t<path>"
    meta, path = line.split("\t")
    mode, typ, sha = meta.split()
    local[path] = (mode, sha)
remote = {t["path"]: (t["mode"], t["sha"]) for t in pushed["tree"] if t["type"] == "blob"}
assert local == remote, "TREE MISMATCH"
print("tree verified byte-identical:", commit["tree"]["sha"])
print("BRANCH_HEAD:", commit["sha"])
