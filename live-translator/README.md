# Live Translator - German → English, live, on your own Mac

Put the Teams / phone meeting on the speaker, start Live Translator on the laptop
next to it, and read along: **every time the speaker pauses, the German line appears
with the English translation underneath** - usually about a second later.
Everything runs locally and free (no DeepL / cloud / API keys), and every meeting
is saved automatically.

```
15:16:02  Wir müssen das Angebot für den Kunden bis Freitag, den 17. Oktober, fertig haben.
          We need to have the offer for the customer ready by Friday, October 17.
```

## 1. Install (once, ~15-30 minutes, mostly downloading)

Open **Terminal** and paste:
```bash
cd ~/Downloads; T=$(ls -t live-translator-*.tar.gz 2>/dev/null | head -1); [ -n "$T" ] && tar -xzf "$T"; bash live-translator/setup-mac.sh
```
(It unpacks the newest `live-translator-*.tar.gz` in Downloads and runs the setup.)
This
1. moves the app to **`~/Documents/LiveTranslator`** (and removes the copy in Downloads),
2. installs it: Python packages, **Ollama** if missing, the speech models (~3 GB: German Whisper +
   the live-preview model) and the translation model (~3 GB), then a self-test that measures
   accuracy and speed on your Mac,
3. puts **`Live Translator`** on your **Desktop**.

*If Homebrew is not installed and Ollama is missing, the installer asks you to install
the Ollama app from <https://ollama.com/download> - then run the setup line again.*
Running the setup again later (e.g. for an update) is safe: meetings are never touched. An update
downloads only what is new (for 1.4: the German speech model and the live-preview model, ~3 GB).

## 2. Use it

**Desktop → double-click `Live Translator` = START. Double-click it again = STOP.**

* START runs the app in the background and opens it in the browser at <http://127.0.0.1:8765>
  (the Terminal window that appears can be closed).
* STOP saves the meeting in progress, stops the app and unloads the translation model, so the
  Mac gets its memory back. (Ollama itself keeps running for your other apps.)
* *The very first time, macOS asks whether Terminal may use the microphone → **Allow**.*

In the browser:
1. Optional: type a name for the meeting in the top bar (you can also do it later).
2. Press **Start** when the meeting begins. The green bar shows that it hears the speaker;
   the blue dot lights up when it detects speech.
3. Press **Stop** at the end of the meeting. It is already saved: the transcript stays on screen
   with a green **Saved** card - type a name there if you like, **Open** it, or click
   **New meeting** to clear the screen for the next one.

| Button | What it does |
| --- | --- |
| **A− / A+** | Text size |
| **DE** | Show / hide the German line (English only) |
| **Meetings** | All saved meetings by date - open, rename, download, delete, "Continue this meeting", "Meeting notes" |
| **⚙** | Settings (microphone, sensitivity, pause length, models, glossary) |

In a narrow window (or on a phone) the header has two rows; text size and the German line are
then in ⚙ Settings → Display.

### Where meetings are saved

```
~/LiveTranslator/Meetings/
    2026-10-08/
        15-15 - Weekly sync with Thomas/
            transcript.md      ← readable: time, German, English
            transcript.jsonl   ← machine-readable, written line by line
            meta.json
            summary.md         ← if you clicked "Meeting notes"
```
A meeting is written to disk **line by line while it happens** - closing the laptop or a crash
loses nothing. The folder is named after the date and start time; give it a name in the top bar
at any time (the folder is renamed). Meetings stay until you delete them. An empty meeting
(Start → Stop with nothing said) is not kept.

## 3. Getting the best accuracy

* **Volume**: set the meeting volume so the bar is mostly in the middle while people talk.
  Quiet/loud swings are handled automatically (automatic gain + per-line levelling), but a
  speaker that is far too quiet or distorted is harder for anyone.
* **Placement**: laptops' microphones are at the top of the screen or next to the keyboard -
  put the speaker within ~50 cm, pointing at it. Avoid the laptop fan blowing at the mic.
* **Glossary (⚙ Settings → Names, products, abbreviations)**: add the names of your boss and
  colleagues, your company/product names and abbreviations (e.g. `Herr Schneider`, `SAP S/4HANA`,
  `Ausländerbehörde`). Both the speech model and the translator use it - this makes the
  biggest difference for names.
* **What is the meeting about?**: one sentence of context ("weekly project meeting about the
  data migration") helps the translator choose the right words.
* **Pause length**: 0.5 s (default) gives the fastest lines. If your boss pauses a lot in the
  middle of sentences and you get many half sentences, raise it to 0.7-0.8 s.
* **Sensitivity**: "High" if the speaker is quiet or far away; "Low" in a noisy room.
* **Speech model**: *German turbo* (default) is Whisper turbo fine-tuned on German speech
  (primeLine) - same speed, clearly fewer errors on German test sets. *Whisper large-v3* is the best
  measured on Bavarian / Franconian / Swabian dialects, but each line takes about twice as long -
  worth trying on a MacBook Pro or when plugged in, if a speaker's dialect is very strong.
* **Translation model**: *Gemma 3 4B* (default) is fast and reliable next to the speech model. Newer
  models (*Gemma 4 E4B*, *TranslateGemma 4B*) can be tried - but nobody has published how well small
  models translate spoken German into English, so compare them on **your own** last meeting:
  ```bash
  ~/Documents/LiveTranslator/.venv/bin/python -m livetranslator compare-llm --pull
  ```
  It translates the saved German lines again with each model, measures the speed on your Mac and
  writes `model-comparison.md` (side by side) into that meeting's folder. Pick the one you like in
  Settings. If a model is too slow during a meeting, the app switches to the fast one by itself and the
  header shows *Translator (fast)*.
* **Dialect**: the translator knows the usual dialect words (*mir* = we, *net/ned* = not, *heuer* =
  this year, *passt scho* = it's fine, *halb drei* = 2:30 …) and checks every line for lost negations
  and changed numbers - if one is missing, it asks itself once more.

## 4. How it works (why it is fast *and* accurate)

1. **Listening**: microphone → resampling → 80 Hz rumble filter → automatic gain →
   **Silero VAD** (neural voice detector) decides speech / not speech every 32 ms.
2. **Lines**: a line ends when the speaker pauses (0.5 s). The moment a pause starts, the line is
   already being transcribed - when the pause turns out to be the end of the line, its German is
   ready at once. If the words so far clearly continue ("…, dass", "und", "für den"), the line is
   held open a little longer, so sentences are not cut at a breath. Long monologues are split at
   ~14 s at the quietest point between words.
3. **Live preview**: while someone speaks, a grey German preview (and after ~4 s a grey English
   preview) shows what is said so far. On Apple Silicon it comes from **Parakeet** - a speech model
   whose cost grows with the length of the audio, so it refreshes often for a fraction of the GPU
   time (Whisper always processes 30 s of audio, even for a 2-second phrase).
4. **German text**: each line is loudness-normalised (quiet words lifted, loud bursts tamed) and
   transcribed by **Whisper turbo fine-tuned on German** on the Mac's GPU (MLX), with your glossary
   and the previous sentence as context. It is converted for the Mac once, on first start. Typical
   Whisper hallucinations ("Untertitel im Auftrag des ZDF", "Vielen Dank." from noise) and
   repetition loops are filtered out.
5. **English**: a local LLM (**Gemma 3 4B** via Ollama)
   translates each line *with the conversation so far as context* - essential for German, where the
   verb and the "nicht" often come at the very end, or in the next line. It knows dialect words,
   keeps numbers, dates, names and negations exact (and double-checks negations and numbers) and
   fixes obvious mis-hearings from context. The English is streamed word by word. If a translation
   fails (Ollama busy, out of memory), the German is saved and the line is retried automatically in
   the background - new lines always go first.
6. **Keeping the Mac cool**: a 1-1.5 h meeting must not make a MacBook (Air) throttle. The app reads
   macOS' thermal state, Low Power Mode, battery and memory pressure, and measures its own speed
   (the translator's words per second) against its first minutes. When the Mac gets warm it slows
   down only the extras - the live
   preview refreshes less often, the English preview pauses - and the header shows *Eco / Cool
   mode*. Every line is still transcribed and translated as usual. (Settings → Performance.)

Measured in the test suite (simulated laptop speaker + room echo + noise + volume swings,
240 sentences, 5 voices): every sentence detected, **no lines from noise/typing/hum**, line closed
~0.6 s after the speaker stops. Real recognition/translation accuracy and speed on *your* Mac:
run the self-test (below).

## 5. Self-test and troubleshooting

```bash
.venv/bin/python -m livetranslator selftest      # accuracy + speed on this Mac (~5 min)
.venv/bin/python -m livetranslator devices       # list microphones
~/Documents/LiveTranslator/toggle.sh mictest      # which microphones really hear sound
~/Documents/LiveTranslator/toggle.sh doctor       # translation health check with timings
```
The self-test speaks German test sentences with the Mac's German voice ("Anna" - if missing:
System Settings → Accessibility → Spoken Content → System Voice → Manage Voices → German),
plays them through simulated rooms (clean / desk / hard) in real time and reports the word error
rate and how long German and English take to appear.

| Problem | Fix |
| --- | --- |
| Won't start | Look at `~/LiveTranslator/logs/app.log`; `~/Documents/LiveTranslator/toggle.sh status` shows whether it runs. |
| No sound / "No sound from '…' - checking the microphones" | Nothing to do: the app re-opens the microphone, tries the Mac's other inputs (built-in mic first) and, if macOS blocks all of them, switches to **this browser's microphone** by itself (click *Allow* if the browser asks). To see which inputs really hear sound: ⚙ Settings → **Test microphones**, or in Terminal `~/Documents/LiveTranslator/toggle.sh mictest`. To use the Mac microphone directly again: System Settings → Privacy & Security → Microphone → Terminal on, quit Terminal (⌘Q), start again. |
| German appears, English does not / is late | Lines now show *translating… N s* and the reason. The app switches to the fast model automatically when the big one is too slow. Run `~/Documents/LiveTranslator/toggle.sh doctor` - it tests Ollama and both models and prints timings (paste it when asking for help). Close apps that use a lot of memory; if you also run LocalLLM with a local model, stop it during meetings. |
| *Translator* pill is red | Start the **Ollama** app. If it says the model is missing: ⚙ Settings → Download model. German is still shown and saved; missing translations are filled in automatically when Ollama is back. |
| *Speech* pill is red | First start needs internet to download the speech model. Check the log: `~/LiveTranslator/logs/server.log`. |
| Lines too slow | Check the self-test numbers. Choose *Gemma 3 4B* in Settings for faster translation. |
| *Cool mode* in the header | The Mac is warm, on battery or short of memory: the live preview is slowed down so the Mac does not throttle. Every line is still translated. Plug in the charger or close heavy apps; Settings → Performance → *Maximum speed* turns it off (except when the Mac is really hot). |
| Too many half sentences | Settings → pause 0.7-0.8 s. |
| Wrong microphone | Settings → Microphone. |

## 6. Later: using it under a host name / on the company laptop

The app is a small web server, so it can be opened from another computer.

**Recommended - the Mac does the work, the company laptop only sends the Teams audio**
(no installation on the company laptop, perfect digital audio instead of speaker → mic):

1. On the Mac: `tools/make_cert.sh <your-mac-name>.local` (browsers only allow audio capture over
   https), then
   ```bash
   ./start.sh --host 0.0.0.0 --token <choose-a-secret> \
              --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
   ```
2. In Settings choose **Audio input → Audio sent from a browser**.
3. On the company laptop open `https://<your-mac-name>.local:8765/?token=<secret>` in Chrome or
   Edge (accept the certificate warning once), press **Start**, then **Send tab / system audio**
   and share the screen with **"Also share system audio"** ticked. The Teams audio now goes
   straight to the translator. (You can read the transcript on either machine.)

**Running it entirely on another computer**: copy the folder, run `install.sh` (macOS/Linux) or
`install.bat` (Windows, needs Python 3.12 from python.org and Ollama), then `start.sh` /
`start.bat`. Without an Apple-Silicon GPU, speech recognition runs on the CPU (faster-whisper) -
fine on a recent laptop, slower on old ones. To give it a fixed name like
`translator.mycompany.local`, point that name at the machine in your DNS/hosts file and use the
`--host 0.0.0.0 --token …` options above.

Always use `--token` when listening on the network: the transcripts are private.

## 7. Files

```
live-translator/
  setup-mac.sh                                       one-time: move to ~/Documents, install, Desktop icon
  toggle.sh / Live Translator.command                start ⇄ stop (what the Desktop icon runs)
  install.sh / start.sh                              install / run in the foreground (macOS / Linux)
  install.bat / start.bat                            Windows
  livetranslator/
    frontend.py   audio in → lines (resample, filter, gain, voice detection, line splitting)
    dsp.py        filters, automatic gain, per-line levelling
    vad.py        Silero VAD (ONNX, bundled)
    segmenter.py  pause / long-monologue logic
    asr.py        Whisper backends (MLX on Apple Silicon, faster-whisper elsewhere), Parakeet preview
    mlx_convert.py  converts German Whisper fine-tunes for the Mac (once)
    governor.py / macsignals.py   keeps the Mac cool (thermal state, battery, speed)
    llmcheck.py   compares translation models on your last meeting (compare-llm)
    textclean.py  hallucination / repetition filter
    translate.py  Ollama translator with conversation context, streaming, meeting notes
    pipeline.py   threads that tie it together
    storage.py    meetings on disk
    server.py     web server + WebSockets
    static/       the web page
  tests/          automated tests (pytest)
  tools/          evaluation + demo helpers
```
Developer checks: `pip install pytest playwright` then `python -m pytest tests`.
