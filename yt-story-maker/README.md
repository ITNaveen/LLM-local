# StoryMaker — your automatic Hindi video editor

Type a **topic** and a short **story description**. StoryMaker searches YouTube, studies the
best footage, writes a five-act Hindi story, records the narration, picks royalty-free music,
cuts everything on the beat and hands you a finished **8–15 minute MP4**, plus a thumbnail,
Hindi subtitles and a ready-to-paste YouTube title, description and tags.

Everything runs **on your Mac, for free**: no paid APIs, no cloud account.

```
Topic:        Virat Kohli West Indies tour 2026
Description:  Kohli under pressure after two low scores, media doubts him, then a
              match-winning century in the final Test. Celebration, crowd, press conference.
                                   ↓
              10-minute cinematic Hindi video, ready to upload
```

---

## Start it (Mac)

1. Install [Homebrew](https://brew.sh) and [Ollama](https://ollama.com) if you don't have them.
2. Pull the free AI models once (you may already have them from LocalLLM):
   ```bash
   ollama pull qwen3.5:9b        # writes the story and Hindi narration (fine on 24 GB RAM)
   ollama pull bge-m3            # understands Hindi + English for matching clips to the story
   ```
   With 48 GB RAM, `ollama pull gemma4:26b` writes noticeably better Hindi. Pick it in **Settings**.
3. Double-click **`StoryMaker.command`**. The first run installs ffmpeg, deno and the Python
   packages. Your browser then opens **http://localhost:7777**.

For burned-in Hindi titles and subtitles, ffmpeg needs text support. If the top bar says
*"ffmpeg: no Hindi text"*, run `brew install ffmpeg-full` once.

## Make a video

1. **Topic**: the big keyword (*Prime Minister Modi Operation Sindoor*).
2. **Story description**: the angle, what should happen, what to show. The clearer this
   is, the better the video.
3. Pick a length (8–15 min), a theme (Epic / Emotional / Documentary / Thriller), and how much
   Hindi narration you want (None / Light / More).
4. Keep **"Let me review the script"** on. When the story is ready you see every scene, its
   clips and the Hindi narration lines. Change any line, then press **Render video**.
5. Download the MP4, thumbnail, `.srt` subtitles and the upload kit.

A 10-minute video takes roughly 15–40 minutes on an M-series Mac, depending on your internet
and model. You can close the tab; the job keeps running, and a failed job resumes from where it
stopped (**Resume**).

## How the "editor brain" works

| Step | What happens |
|---|---|
| 1. Research | The AI turns your topic into 10–16 searches (Hindi + English) and reads up to **500 videos' titles, views and channels**. Downloads nothing. |
| 2. Rank | Scores each video for relevance, popularity and footage quality. Reaction videos, Shorts and podcasts are pushed down. The AI then picks the best ~24. |
| 3. Moments | Reads each video's **transcript** (what is said, when) and YouTube's **"Most replayed" graph** (the seconds people rewatch). These become hundreds of candidate moments. |
| 4. Story | The AI writes a **five-act story**: opening hook → build-up → rising tension → climax → ending. For every beat it decides **who speaks**: the *narrator* (Hindi voice-over), the *clip itself* (speech, commentary, crowd) or *music only* (montage). |
| 5. Clip choice | Every beat gets the best matching seconds: whole sentences for dialogue, the hottest replayed seconds for the climax, and no footage reused. It opens with a **cold-open teaser** of the biggest moments. |
| 6. Sound | Narration is recorded (free Microsoft Edge Hindi neural voice; offline Piper or the Mac voice as backup). Music ducks automatically under speech. Cuts in montages **snap to the music's beat**. |
| 7. Render | Downloads **only the seconds it uses**, fills vertical clips with a blurred background, adds a light colour grade, a title card, Hindi subtitles, YouTube loudness (−14 LUFS) and chapters. |

## Teach it your favourite editing style

Paste a video whose editing you love (e.g. the Operation Sindoor edit,
`https://www.youtube.com/watch?v=hT-hQZCl6Og`) into **Learn an editing style** and name it.
StoryMaker measures how long the shots are in each part of the video, how the pace speeds up
into the climax, and how much of it is dialogue. Pick that style for your next video to copy
the **rhythm**, never the footage.

## Music

Put royalty-free tracks into `music/<mood>/` (`epic`, `tense`, `emotional`, `triumphant`, `calm`,
`dark`). See [`music/README.md`](music/README.md). The **YouTube Audio Library** is the safest
source. With an empty library a simple generated score is used. It works, but real tracks
sound far better.

## Copyright: please read

StoryMaker credits every source channel in the description and adds the usual fair-use note.
The safest music to use is from the YouTube Audio Library. However, **re-using other channels'
footage can still get Content ID claims or strikes**. Voice-over on top does not stop the
system from matching the footage. Prefer official / news / government footage, keep your
narration and storytelling substantial (that is what makes a video *transformative*), and
avoid film clips and songs. The responsibility for what you upload stays with you.

## Settings worth knowing

* **Story model**: any Ollama model; bigger = better Hindi.
* **Thinking mode**: smarter stories, several times slower.
* **YouTube cookies from browser**: only if YouTube starts asking you to sign in.
* **Voice**: Male (Madhur) or Female (Swara) per video; speed and pitch in Settings.
  For a fully offline voice, install `pip install piper-tts`, download a Hindi `.onnx` voice
  and set its path.

## Troubleshooting

| Problem | Fix |
|---|---|
| "YouTube search returned nothing" / sign-in errors | Restart StoryMaker (it updates yt-dlp), make sure `deno` is installed, or set *cookies from browser* in Settings. |
| Narration silent | The Edge voice needs internet; otherwise set the voice engine to *macOS voice* or Piper. |
| Story looks generic | Ollama is not running or the model isn't installed (top bar shows it). Write a more detailed description. |
| No Hindi text in the video | `brew install ffmpeg-full` |

Try **Demo mode** (no YouTube, synthetic clips) to check the whole pipeline works on your machine.

## For developers

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 40+ tests incl. full renders on synthetic footage
python -m storymaker         # start the web app
```

Code map: `storymaker/research.py` (search + ranking) → `moments.py` (transcripts + replay
graph) → `story.py` (five-act plan + clip choice) → `voice.py` → `music.py` → `timeline.py` →
`render.py` → `publish.py`, orchestrated by `pipeline.py` and served by `web.py`.
