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

## The emotional Hindi voice (recommended)

The built-in Microsoft voice is free but flat. For a narrator with real emotion, double-click
**`install-emotional-voice.command`** once. It installs AI4Bharat's **Indic Parler-TTS**:
free, offline, made for Indian languages, and it can be told *how* to speak. It downloads
about 5 GB and plays a test line at the end.

The model is free, but its makers ask everyone to accept their licence once. The installer
walks you through it:
1. It opens the model page: log in or sign up at Hugging Face (free) and click **"Agree and access
   repository"**.
2. It opens the token page: click **Create token**, copy it, and paste it into the installer.

StoryMaker then uses the voice automatically; the top bar shows **emotional voice**. If it shows
**emotional voice LOCKED**, run the installer again. Every line gets its own emotion: angry for
lines with "!", urgent for questions. In **Settings** you can pick the speaker (Rohit, Aman,
Divya, Rani) or describe your own style, e.g. *"Rohit speaks angrily and fast, like a fiery TV
debate"*.

## Make a video

1. **Topic**: the big keyword (*Prime Minister Modi Operation Sindoor*).
2. **Story description**: the angle, what should happen, what to show. The clearer this
   is, the better the video.
3. Optional: **Your scene outline**. Write the story beats yourself, one per line, and the AI
   follows them in that order.
4. Pick a length (8–15 min), a theme (default **🔥 Hindi news**), and how much narration you
   want: *None* (headlines only), *Light* (about 5 lines), or *More* (about 8 lines). The
   narrator never talks over the clips; the people in the footage carry the story.
5. Press **Make my video** and walk away: it runs start to finish on its own (the Mac stays
   awake while StoryMaker runs). Optional: tick **"Pause before rendering"** to check the
   script first; then you see every scene, its
   clips and the Hindi narration lines. Change any line, then press **Render video**.
6. Download the MP4, thumbnail, `.srt` subtitles and the upload kit.

A 10-minute video takes roughly 15–40 minutes on an M-series Mac, depending on your internet
and model. You can close the tab; the job keeps running, and a failed job resumes from where it
stopped (**Resume**).

## How the "editor brain" works

StoryMaker edits like a Hindi news channel's video desk: real people in the footage tell the
story, and the narrator only sets the base.

| Step | What happens |
|---|---|
| 1. Research | The AI turns your topic into 12–16 searches (Hindi, Hinglish, English) aimed at the drama: clashes, लाठीचार्ज, पथराव, families, angry statements, viral clips, and the big Hindi channels' coverage. It reads up to **500 videos' titles, views and channels** and downloads nothing. |
| 2. Screen | Every shortlisted video is checked. Any angle of the same story counts (backstory, victims, police, courts, raw viral videos). **The audience only hears Hindi**: English speakers, even when their captions are in Hindi letters, are left out or briefly retold by the narrator (max 3 per film). Regional languages are never used. |
| 3. Understand | Every transcript is cut into **passages** (complete thoughts of 10–35 s). The AI reads each one: who says what, the emotion, how strong it is, and its single most gripping line. |
| 4. Plan scenes | Five acts that escalate: buildup → rising anger → explosive climax → ending. The **Hindi clips play back-to-back**. The narrator speaks only to set the base as an act opens, for one reveal at most, and for the closing question. It never describes the picture (it can't see it) and never adds facts. |
| 5. Editor review | A second AI pass cuts and reorders like a senior editor and scores the film. |
| 6. Fact check | Every narrator line is checked against what the footage says. Lines with numbers or big names (PM, Supreme Court, …) that no source mentions are cut. |
| 7. Cold open | The first 15 seconds: 3–4 of the most gripping Hindi soundbites from different channels, with flash-cuts, punch-ins and hits, then the narrator's hook question and a **title sting** with a boom. |
| 8. Cut | Long soundbites get **cutaways**: the speaker is seen first, then their voice continues over clashes, crowds and barricades, then back to them. Pain and grief stay on the face. |
| 9. Sound & render | A built-in news-thriller score (or your tracks) is ducked under speech, with whooshes into every chapter and a riser into the climax. Big Hindi headlines and subtitles are drawn with any ffmpeg. 1080p, chapters, YouTube loudness. |

In the **review** (optional) you see every scene with what is actually said, why it is there,
and a **Keep** box. Untick anything you don't like, edit any text, then press **Render video**.

## Teach it your favourite editing style

Paste a video whose editing you love (e.g. the Operation Sindoor edit,
`https://www.youtube.com/watch?v=hT-hQZCl6Og`) into **Learn an editing style** and name it.
StoryMaker measures how long the shots are in each part of the video, how the pace speeds up
into the climax, and how much of it is dialogue. Pick that style for your next video to copy
the **rhythm**, never the footage.

## Music

With an empty music folder StoryMaker uses its **built-in news-thriller score**: a dark drone,
ticking pulse, heartbeat and low hits, generated on your Mac, so it is copyright-free. Real
tracks sound even better. Put royalty-free tracks into `music/<mood>/` (`epic`, `tense`,
`emotional`, `triumphant`, `calm`, `dark`); see [`music/README.md`](music/README.md). The
**YouTube Audio Library** is the safest source.

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
| No Hindi text in the video | Restart StoryMaker (it installs the Pillow package that draws Hindi text). |
| Top bar: "emotional voice LOCKED" | Run `install-emotional-voice.command` again and follow its 2 steps. |

Try **Demo mode** (no YouTube, synthetic clips) to check the whole pipeline works on your machine.

## For developers

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 100+ tests incl. full renders on synthetic footage
python -m storymaker         # start the web app
```

Code map: `storymaker/research.py` (search + ranking) → `moments.py` (transcripts + replay
graph) → `story.py` (five-act plan + clip choice) → `voice.py` → `music.py` → `timeline.py` →
`render.py` → `publish.py`, orchestrated by `pipeline.py` and served by `web.py`.
