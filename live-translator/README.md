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

## 1. Install (once, ~15 minutes, mostly downloading)

1. Copy the `live-translator` folder somewhere permanent, e.g. `~/Documents/LiveTranslator`.
2. Open **Terminal** in that folder and run:
   ```bash
   ./install.sh
   ```
   It sets up Python, installs the packages, installs **Ollama** (if missing), downloads the
   speech model (~1.6 GB) and the translation model (~8 GB), and finishes with a self-test
   that measures accuracy and speed on your Mac.

   *If Homebrew is not installed and Ollama is missing, the installer asks you to install
   the Ollama app from <https://ollama.com/download> - then run `./install.sh` again.*

## 2. Use it

1. Double-click **`Live Translator.command`** (or run `./start.sh`). The browser opens at
   <http://127.0.0.1:8765>. Keep the Terminal window open while you use it.
   *The very first time, macOS asks whether Terminal may use the microphone → **Allow**.*
2. Optional: type a name for the meeting in the top bar (you can also do it later).
3. Press **Start** when the meeting begins. The green bar shows that it hears the speaker;
   the blue dot lights up when it detects speech.
4. Press **Stop** at the end. The meeting is already saved.

| Button | What it does |
| --- | --- |
| **A− / A+** | Text size |
| **DE** | Show / hide the German line (English only) |
| **Meetings** | All saved meetings by date - open, rename, download, delete, "Continue this meeting", "Meeting notes" |
| **⚙** | Settings (microphone, sensitivity, pause length, models, glossary) |

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
* **Speech model**: *large-v3-turbo* (default) is fast and very good. *large-v3* is slightly more
  accurate but slower - try it if your Mac keeps up (the self-test tells you).
* **Translation model**: *Gemma 3 12B* (default, best). *Gemma 3 4B* is ~3× faster with slightly
  lower quality. Any other Ollama model can be chosen too.

## 4. How it works (why it is fast *and* accurate)

1. **Listening**: microphone → resampling → 80 Hz rumble filter → automatic gain →
   **Silero VAD** (neural voice detector) decides speech / not speech every 32 ms.
2. **Lines**: a line ends as soon as the speaker pauses (0.5 s). Long monologues are split
   regularly (shorter pauses are accepted the longer a line gets, and at ~20 s it cuts at the
   quietest point between words) so you never wait long.
3. **German text**: each line is loudness-normalised (quiet words lifted, loud bursts tamed)
   and transcribed by **Whisper large-v3-turbo** on the Mac's GPU (MLX), with your glossary and
   the previous sentence as context. Typical Whisper hallucinations ("Untertitel im Auftrag des
   ZDF" etc.) and repetition loops are filtered out.
4. **English**: a local LLM (**Gemma 3 12B** via Ollama) translates each line *with the
   conversation so far as context* - essential for German, where the verb and the "nicht" often
   come at the very end, or in the next line. It is told to keep numbers, dates, names and
   negations exact and to fix obvious mis-hearings from context. The English is streamed word by
   word. Speech recognition of the next line runs while the previous one is being translated.

Measured in the test suite (simulated laptop speaker + room echo + noise + volume swings,
240 sentences, 5 voices): every sentence detected, **no lines from noise/typing/hum**, line closed
~0.6 s after the speaker stops. Real recognition/translation accuracy and speed on *your* Mac:
run the self-test (below).

## 5. Self-test and troubleshooting

```bash
.venv/bin/python -m livetranslator selftest      # accuracy + speed on this Mac (~5 min)
.venv/bin/python -m livetranslator devices       # list microphones
```
The self-test speaks German test sentences with the Mac's German voice ("Anna" - if missing:
System Settings → Accessibility → Spoken Content → System Voice → Manage Voices → German),
plays them through simulated rooms (clean / desk / hard) in real time and reports the word error
rate and how long German and English take to appear.

| Problem | Fix |
| --- | --- |
| "The microphone is completely silent" | System Settings → Privacy & Security → Microphone → enable **Terminal**, then restart. |
| *Translator* pill is red | Start the **Ollama** app. If it says the model is missing: ⚙ Settings → Download model. German is still shown and saved; missing translations are filled in automatically when Ollama is back. |
| *Speech* pill is red | First start needs internet to download the speech model. Check the log: `~/LiveTranslator/logs/server.log`. |
| Lines too slow | Check the self-test numbers. Choose *Gemma 3 4B* in Settings for faster translation. |
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
  install.sh / start.sh / Live Translator.command   macOS / Linux
  install.bat / start.bat                            Windows
  livetranslator/
    frontend.py   audio in → lines (resample, filter, gain, voice detection, line splitting)
    dsp.py        filters, automatic gain, per-line levelling
    vad.py        Silero VAD (ONNX, bundled)
    segmenter.py  pause / long-monologue logic
    asr.py        Whisper backends (MLX on Apple Silicon, faster-whisper elsewhere)
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
