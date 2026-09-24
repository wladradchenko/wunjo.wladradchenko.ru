# Worked examples

Four archetypes covering the axes that matter: `kind` (local vs api) and
`target` (video/audio/face/generator). Each `main.py` follows the stdout
protocol from [authoring-guide.md](authoring-guide.md). The `stub-*` folders next
to this `skills/` dir are runnable versions of the first three.

A shared preamble every `main.py` uses:

```python
import argparse, json, os, sys

def emit(line): sys.stdout.write(line + "\n"); sys.stdout.flush()

def load_job():
    p = argparse.ArgumentParser(); p.add_argument("--job", required=True)
    with open(p.parse_args().job, encoding="utf-8") as f: return json.load(f)
```

## A. Local, `target: video` — process each selected clip

`plugin.json` (excerpt): `"kind":"local"`, `"venv":"private"`, `"target":"video"`,
`"input":{"clip":"video","multiple":true}`, `"result":{"type":"video","place":"bin"}`,
`params` for tunables. `requirements.txt`: your torch/onnx/etc. deps.

```python
def main():
    job = load_job()
    clips = job["input"]["clips"]
    params = job.get("params", {})
    outputs = []
    for i, clip in enumerate(clips):
        emit(f"info:processing {clip['path']}")
        out = os.path.join(job["output_dir"], f"out_{i}.mp4")
        # ... run the model on clip['path'][clip['in']:clip['out']] → out ...
        outputs.append({"type": "video", "path": out})
        emit(f"progress:{int((i+1)*100/len(clips))}")
    emit("result:" + json.dumps({"outputs": outputs}))
    return 0

if __name__ == "__main__": sys.exit(main())
```

## B. Local, `target: audio`, with a downloaded model

`plugin.json`: `"target":"audio"`, `"models":[{"name":"sep.onnx","url":"https://…","sha256":"…","size_mb":40}]`,
`"result":{"type":"audio","place":"bin"}`.

```python
def main():
    job = load_job()
    model = os.path.join(os.path.dirname(__file__), "models", "sep.onnx")
    if not os.path.exists(model):
        emit('need:{"kind":"model","name":"sep.onnx"}'); return 4
    clip = job["input"]["clips"][0]
    out = os.path.join(job["output_dir"], "vocals.wav")
    emit("progress:10")
    # ... onnxruntime.InferenceSession(model) → separate clip['path'] → out ...
    emit("progress:100")
    emit("result:" + json.dumps({"outputs": [{"type": "audio", "path": out}]}))
    return 0
```

## C. API, `target: generator` — text → audio via a paid provider

`plugin.json`: `"kind":"api"`, `"target":"generator"`,
`"provider":{"name":"elevenlabs","key_setting":"WUNJO_KEY_ELEVENLABS","signup_url":"https://elevenlabs.io"}`,
`"input":{"clip":"none"}`, `"result":{"type":"audio","place":"bin"}`,
`params` for the text and voice. `requirements.txt`: `httpx` (light ⇒ `venv` may
be `shared`).

```python
def main():
    job = load_job()
    key = os.environ.get("WUNJO_KEY_ELEVENLABS")
    if not key:
        emit('need:{"kind":"api_key","provider":"elevenlabs"}'); return 3
    text = job["params"].get("text", "")
    out = os.path.join(job["output_dir"], "speech.mp3")
    emit("info:calling ElevenLabs")
    # import httpx; r = httpx.post(url, headers={"xi-api-key": key}, json={...})
    # open(out, "wb").write(r.content)
    emit("result:" + json.dumps({"outputs": [{"type": "audio", "path": out}]}))
    return 0
```

Note: the key is read from the environment only; it is never in `job.json`,
argv, or any output line.

## D. `target: face` — act on one detected face

`plugin.json`: `"target":"face"`, `"input":{"clip":"video"}`,
`"result":{"type":"video","place":"bin"}` (or `none` to just report).

```python
def main():
    job = load_job()
    face = job["input"].get("face", {})
    rect, frame = face.get("rect"), face.get("position")   # [x,y,w,h] in 0..1
    clip = job["input"]["clips"][0]
    emit(f"info:face at {rect} on frame {frame}")
    # ... blur / replace / retouch that region across the clip → out ...
    emit("result:" + json.dumps({"outputs": [], "message": "done"}))
    return 0
```

## Packaging & testing

```
python ../pack.py --check my-plugin      # validate the manifest
python ../pack.py my-plugin              # → dist/my-plugin-<version>.wmplugin
python my-plugin/main.py --job /tmp/job.json   # dry-run against a hand-written job
```

Then in the app: Settings ▸ Plugins ▸ Load Plugins ▸ From archive/folder, review
the metadata, Import; the plugin gets its own tab where its environment, models
and key are set up before first use.
