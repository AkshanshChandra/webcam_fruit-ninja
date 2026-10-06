# Fruit Ninja: Webcam Edition

Slice flying fruit with your index finger in front of a webcam. Built with OpenCV
(camera and rendering) and MediaPipe Hands (fingertip tracking).

## Setup

Requires Python 3.9–3.13 and a webcam.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The hand-tracking model ships in `models/hand_landmarker.task`, so the game works
offline. If the file is missing it is downloaded once on first launch.

**macOS:** the first launch asks for camera permission for your terminal (or IDE).
If you dismissed the prompt, enable it under *System Settings → Privacy & Security
→ Camera*, then restart the terminal.

### Command-line options

| Option | Effect |
|---|---|
| `--windowed` | Open a large 1280×720 window instead of fullscreen |
| `--camera N` | Use webcam N (try `1` if the wrong camera opens) |
| `--mute` | Start with sound off |
| `--no-audio` | Don't open the audio device at all |
| `--debug` | Show an FPS counter |

## Controls

| Where | Key | Action |
|---|---|---|
| Menu | ↑ / ↓ (or W / S) | Choose Survival / 60s Time Attack / Leaderboard |
| Menu | SPACE or ENTER | Start / open the selected item |
| Menu | L | Open the leaderboard |
| Menu | ESC twice | Quit |
| Game | *index finger* | Swipe fast through fruit to slice it |
| Game | P | Pause / resume |
| Game | ESC | Pause (ESC again while paused returns to the menu) |
| Game over | type letters | Enter your name; ENTER saves, ESC skips |
| Leaderboard | SPACE | Back to the menu |
| Anywhere | M | Mute / unmute |
| Anywhere | F | Toggle fullscreen |
| Anywhere | D | Toggle the FPS counter |
| Anywhere | hold left mouse button | Use the mouse as the blade (backup if tracking struggles) |

## Rules

- **Survival:** 3 lives (shown as apples, top left). Every fruit that falls off the
  bottom unsliced costs a life. Slicing a bomb ends the round immediately.
  Fruit comes faster, in bigger waves, and with more bombs the longer you last.
- **60s Time Attack:** slice as much as you can in 60 seconds. Dropped fruit costs
  nothing, and a bomb costs 50 points instead of ending the round. To make bombs end
  the round in this mode too, set `TIMER_BOMB_ENDS_ROUND = True` in `config.py`.
- **Scoring:** +10 per fruit. Slicing fruits less than 0.5s apart builds a combo:
  DOUBLE (+10 bonus), TRIPLE (+30), QUAD (+60), and so on.
- Slices need a real swipe. A fingertip resting on a fruit, or moving slowly through
  it, does not cut it. The hitbox is 30% larger than the fruit you see, which makes
  up for imprecise tracking.

## Running it at an event

- **Attract mode:** when nobody is playing, the menu keeps launching fruit and a
  "ghost blade" slices it. After 25 idle seconds the menu switches to the
  leaderboard, and after 15 more it switches back.
- **Leaderboard safety:** `leaderboard.json` keeps the top 20 scores. Each write
  goes to a temporary file that is fsync'd and then atomically renamed, so a crash
  or power cut can't corrupt it. A score is saved the moment the game-over screen
  appears, under the placeholder name `PLAYER`, and renamed when the player types
  a name. If nobody touches the keyboard for 60 seconds, the score stays saved as
  `PLAYER` and the game moves on. If `leaderboard.json` is ever unreadable, it is
  moved aside as `leaderboard.json.corrupt-<time>` and never deleted.
- **Errors:** if something unexpected goes wrong during a frame, the error is written
  to `crash.log` and the game returns to the menu instead of exiting. If the camera
  is missing or blocked, an on-screen message explains how to fix it (R retries).
  If the camera disconnects mid-game, the game reconnects automatically.
- Most tuning lives in `config.py`: swipe speed threshold, hitbox size, gravity,
  difficulty ramp, idle timers, title text and colors.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Entry point: window, keyboard/mouse input, camera error screen, crash logging |
| `app.py` | Shared game context and the per-frame pipeline (mirror → track → vignette → state) |
| `game_states.py` | State machine: Menu, Countdown, Play, GameOver, Leaderboard; shared `World` simulation |
| `fruit.py` | Procedurally drawn fruit and bomb sprites, physics, slicing into halves, wave spawner |
| `particles.py` | Juice droplets, splats, shockwaves, explosion, animated popup text, flash/shake |
| `hand_tracker.py` | Async MediaPipe hand landmarker plus the `Blade` (glowing trail and swipe detection) |
| `camera.py` | Threaded webcam capture with reconnect and clear permission errors |
| `audio.py` | Synthesized sound effects mixed in real time (no audio files needed) |
| `leaderboard.py` | Crash-safe JSON leaderboard |
| `render.py` | Cached outlined/shadowed text (PIL fonts), alpha blitting, panels, vignette |
| `config.py` | All tunable constants |
