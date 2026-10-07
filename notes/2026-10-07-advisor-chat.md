# Notes from the advisor chat (2026-10-05 to 2026-10-07)

The developer's notes from a separate review chat with Claude (outside Claude Code, read-only reviewer role). Compressed on purpose: context on where the thinking is going, not results. Results live in the README; nothing here changes a bar or a result.

## Where the evidence stands (as of experiment 27's bar)

- On real telecom data, learned words don't replace raw data (experiments 20–25). Two-part messages (learned meaning + exact precision characters) came within 7–9% (experiment 26), but byte for byte a plain code of recent exact values does as well (experiment 26b). The precision part did the work.
- Experiment 27 (next-day, against a strong equal-bytes raw code) is the final telecom test of the current form. The write-up goes ahead whatever it shows.
- One consistent pattern across results: **context helps where data is thin, and fades where data is rich.** Fashion-MNIST: words won at 50 labels, not at thousands. Telecom: weeks of history per square, raw data won. Traffic: sensors with months of history barely used their neighbours. Candidate thesis: *a shared context language pays off where data is thin (new points, few labels, low cost), which is where most real-world cost is.*

## How the vision got sharper in the chat

- **Not a replacement for raw data.** The developer's view: raw data stays; the language is shared context on top of it, so repeated feature engineering isn't needed per model. "A shared dictionary that compresses information automatically", like two data scientists who just say "XGBoost" because they share the background. **Never tested on telecom:** raw data + words vs raw data alone.
- **Context language.** A value only means something in context (500 calls: normal at noon in a business district, alarming at 3 a.m. in a residential square). The predictive words already had this shape: word 1 = what's normal here and now, corrections = how today differs. Relatives that already exist: seasonal-baseline features, contextual anomaly detection. The possibly new part: context learned once, shared and reused, instead of hand-built per model.
- **A generic feature alphabet probably needs three things:** exact symbols (values that must be precise), learned symbols (concepts, type of day) and rules for combining them. Experiment 26 hinted at the first two.
- **The LLM is a translator, not a gap filler.** Experiment 19b: its answers and confidence came from the word cards. It can explain and connect models; it can't recover information the words blurred.
- **Two kinds of sharing, kept apart.** A shared dictionary (the same words mean the same thing everywhere) works. Points learning from neighbours' data failed on traffic (experiment 17, 3.5%). The one open case: a brand-new tower with almost no history, where neighbours' context may be all it has.
- **Universality, later and in layers:** shared basic shapes (level, trend, daily/weekly rhythm, spike, change: the "radicals"), domain vocabularies on top, bridges between dictionaries. Cheapest first test once one domain works: freeze the telecom dictionary and use it on PEMS-BAY. Relatives: Chronos and TimesFM already build universal tokenizers for values; the possibly new layer is shared concept words with measured cards.
- **Science framing (agreed):** the write-up is a checkpoint, not the end. Failures here are about this form, on this data, with this encoder. Know exactly why each one failed, so the idea can come back when the missing piece shows up (EUV lithography and neural networks both did).

## Reviewer habits to keep

- Measure the ceiling before polishing: the experiment 24 audit showed the encoder capped everything, so tuning the words could never reach the bar.
- Any compactness claim gets an equal-bytes raw control (experiment 26b), and the raw control must be the *best* simple code for the question (e.g. the same hour yesterday for next-day forecasts).
- The next project-level bar needs a question the dictionary was never trained toward, and a neural net on raw data as a control (so a pass shows the language, not just a better forecaster).
- Label hindsight as hindsight: "next-hour forecasting favoured exact values" is a lesson learned, not how the test was chosen.
- When something fails, name the most promising workaround and the test that would check it, before moving on. No made-up numbers.
- Be able to explain every experiment and its code yourself (the viva test), even when Claude Code wrote it.

## Candidate next tests (each needs its own bar, committed first)

1. **Raw + words vs raw alone**, on several questions (next hour, next day, unusual activity, one the dictionary was never trained toward), with a third arm of hand-built baseline features (deviation from the usual for this square and hour). The direct test of "context on top of raw data".
2. **Confidence-based fallback:** answer from the words when confident, fetch raw data only when unsure; measure how much raw data is saved at the same accuracy (the "cheaper" claim).
3. **New towers:** squares given only one day of history, their own raw data vs their own raw data + neighbours' words. Write down first why it should work: a new tower has no history, so its context must come from somewhere.
4. **Transfer (later):** the frozen telecom dictionary used on PEMS-BAY.

Order: finish experiment 27 and the write-up first; then at most one of these at a time, with the December 15 checkpoint in mind.
