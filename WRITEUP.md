# A learned data language: what 30 pre-registered experiments showed

*7–9 October 2026, after experiment 27, the final telecom test (its bar was committed before it ran). A checkpoint, not an end.*

## The idea in one paragraph

Turning raw data into features is slow human work, done again for every pair of data and model. This project asked whether a model could do it once, instead: squeeze raw data a person can't read (pixels, sensor streams, telecom traffic) into a few discrete "words" from a shared, frozen dictionary, so that other models, and an LLM, reuse those words instead of relearning everything from the raw data. The closest analogy is a compiler's shared intermediate language, but learned and lossy on purpose, closer to a video codec that keeps what matters. The claim to test: **features learned by one model, stored in compact form, make a second model better or cheaper than starting from raw data.**

## How it was tested

- **Public data only:** IBM Telco churn (warm-up), Fashion-MNIST (a toy, used to develop the recipe), PEMS-BAY road traffic (325 connected sensors) and Telecom Italia (Milan, 10,000 grid squares, hourly SMS, calls and internet).
- **Every success bar was written down and committed before its experiment ran,** so results couldn't be rationalised afterwards. Git shows the order from experiment 19 on.
- **Since experiment 16, a bar only counts as passed if the whole 95% confidence interval clears it;** anything else is "too close to call" and counts as not shown. Telecom intervals resample whole spatial blocks, because neighbouring squares behave alike.
- **One project-level bar** (set on 6 October 2026, before any telecom result) said when the proof of concept as a whole would be proven, with a stop rule for when it failed.
- **A dictionary health gate** (added after a training breakdown in experiment 21): no test is computed on a dictionary whose training didn't settle.
- **Hardware:** a CPU laptop capped at 30% of its cores. Every experiment runs in minutes.

## What worked

- **Compression.** An image becomes 8 bytes instead of 784; a telecom square's day becomes 16–19 bytes instead of 288. At the same size, learned words keep 1.4–2.5x more of a picture than PCA.
- **Learning from few labels, on images.** Learned words beat raw pixels with 50 labels: +4.0 points [+3.1, +4.9], held on a retraining on a second laptop (+3.9 [+3.0, +4.8]).
- **Words that carry meaning on their own.** When words can't lean on position (an unordered set of 8), each learns a whole-garment concept: purity rose from 17% to 70%, and a learned alphabet found 16 "radicals" that look like real concepts (trouser legs, a bag handle, a sneaker sole).
- **A shared language is better learned than reinvented.** A new model trained to write in a frozen dictionary agreed with the original speaker far more often (58–69%) than an independently trained one (47–51%).
- **The language can be stored and read.** A hash-verified SQLite library gives back exactly what was stored. An off-the-shelf LLM reading only words and their computed word cards named the class 78.6% of the time, with useful confidence (95% right when confident, 58% when not), for $2.60.

## What didn't

- **On tidy tables there's nothing to gain** (Telco): the model already finds the patterns in the raw columns.
- **Words of words lost meaning** (experiment 6), and words trained to predict a hidden half learned real knowledge that no reader could get out of the words (experiments 8–9).
- **Connected points added little.** On road traffic, neighbours' raw data improved forecasts by at most 3.5% with three different forecasters; a better forecaster per sensor mattered far more.
- **The LLM adds sentences, not judgement.** It picked the simple card vote's answer 97.6% of the time, and its confidence was the vote's own margin. The meaning lives in the dictionary and its cards.
- **On real telecom data, learned words alone lost to raw data on every question.** The project-level test (experiment 20) was a clear fail: 1.8–2.5x raw data's next-hour error even with all the data. An apparent edge on unusual activity turned out to be calibration (experiment 20b, AUC 0.780 vs 0.834).

## The telecom chain (experiments 20–27)

After the clear fail, each step was a single change with its own bar, and every failure was reviewed before anything new was built.

| Step | What changed | Next-hour error vs raw data (internet / calls) |
|---|---|---|
| 20 | 8 learned set words | +152% / +77% |
| 21–22b | Coarse-to-fine correction words; the training broke until the "revive" step was removed | (training failures; a health gate added) |
| 23 | 16 correction words, healthy dictionary | +121% / +60% |
| 24 | Audit: the loss is in the encoder's squeeze, before the words | (diagnosis) |
| 25 | Encoder trained to predict the next 6 hours | +115% / +48% |
| 26 | Two-part language: 16 meaning characters + 3 precision characters (the last hour's exact values, 1 byte each) | **+9.4% / +7.1%** (too close to call) |
| 26b | Same bytes spent on the last 6 hours' exact values instead | the two-part message and raw-18 tie: +0.2% / −0.9% |
| 27 | Next-day forecasting (same hour tomorrow) against a strong 18-byte raw code | the raw code wins: two-part message +7.0% / +5.8% worse than it |

**The finding of this chain:** a compact learned summary keeps a day's shape and type but blurs exact values, and next-hour forecasting lives on exact recent values. Adding characters that state exact values closed almost the whole gap, but byte for byte (experiment 26b), spending the same bytes on more exact recent hours does just as well. On next-hour forecasting, the precision part does the work and the learned part adds nothing.

**A caution about hindsight.** It's tempting to say next-hour forecasting was simply the wrong test for a learned summary. That is hindsight. Those questions were chosen deliberately for the project-level bar on 6 October 2026. What can fairly be said is that the risk was written down before the run (the expectation for experiment 20 said words that blur the exact last value may lose on next-hour forecasting), and that **we learned the test favoured exact values**. It was not always a bad test, and calling it one after the fact would be moving the goalposts.

## The final test (experiment 27)

Experiment 26b left one honest question: do the learned characters carry something exact recent values don't, on a question where a day's type should matter? Experiment 27 asks it once, for the same hour tomorrow, against the strongest simple code of the same size (the target's same hour yesterday and two days before, plus neighbouring hours, 18 bytes). The dictionary was trained to predict the next 6 hours, never the next day, so this is also a real reuse test.

The decision rule, committed with the bar: **"If A doesn't beat B on next-day forecasting, the telecom chapter closes: the write-up's conclusion is that learned summaries keep a day's shape but don't beat equal-bytes raw codes on telecom forecasting. No further narrowing before the write-up exists."**

**Result (9 October 2026): not a pass, so the telecom chapter closes.** The strong raw code beat the two-part message on both questions, by 7.0% [6.3, 7.7] on internet and 5.8% [5.1, 6.6] on calls, and came within about 2% of all 288 bytes of raw data (+2.5% / +1.9%). The learned characters did carry real information: the message beat "same hour yesterday" alone by 6–10%. But spending the same bytes on the right exact values did better.

## What this adds up to

**On telecom forecasting, learned summaries keep a day's shape but don't beat equal-bytes raw codes.** That held on next-hour and next-day questions, on a dictionary trained to rebuild and on one trained to predict, with and without precision characters. The lessons:

1. **Learned discrete codes compress well and can carry meaning,** especially when each word has to stand alone, and they help most where labels are scarce (images).
2. **A learned summary isn't a substitute for exact values.** Where a task depends on the precise recent level, the summary's blur is a hard limit that better training didn't move (experiments 24–25). A useful data language needs characters that state exact values as well as characters that describe kind, but on next-hour forecasting a simple fixed code for the values was enough on its own.
3. **The value is in the dictionary, not the reader.** Word cards measured from data gave the meaning and the confidence; the LLM put them into sentences.
4. **The fair comparison is byte for byte.** Comparing a 19-byte message with 288 bytes of raw data hid the real question; the right control is the best simple code of the same size.
5. **Process lessons:** pre-registered bars and whole-interval judgement kept several small "wins" from being over-read (three earlier passes turned out to be within noise); a dictionary needs a health gate before anything is built on it; and a short smoke run can't catch a training breakdown that only appears after several passes.

## What stays open

This closes the telecom chapter **for this form of the idea**: words used *instead of* raw data, from this encoder, on this data. It doesn't close the idea. A pattern runs through every result: shared context helped where data was thin (50 labels on images) and faded where data was rich (weeks of history per telecom square, months per road sensor). The untested versions follow from that, each needing its own bar committed first, one at a time (details in `notes/2026-10-07-advisor-chat.md`):

1. **Words on top of raw data, not instead of it,** on several questions including one the dictionary was never trained toward, with hand-built baseline features as a third arm and a neural net on raw data as a control.
2. **Answer from the words when confident, fetch raw data only when unsure:** how much raw data is saved at the same accuracy (the "cheaper" claim).
3. **New towers with almost no history,** where shared context may be all a point has.
4. **Transfer:** the frozen telecom dictionary reused on road traffic.

The 15 December 2026 checkpoint stands: if nothing on real data has passed by then, the claim gets rethought again.

## Limits

- One telecom dataset (two training weeks, one test week, one city) and one road-traffic dataset; Fashion-MNIST is a toy.
- One forecaster family (LightGBM) for the telecom tests.
- Effects on images are small (a few points).
- The "cheaper" side of the claim (one build cost, then cheap reuse) is still theory: its break-even was never measured, and the searchable library file is 14x bigger than the words in it.
- Untested: the many-tasks reuse the idea is ultimately for (replacing feature engineering across many questions on the same data), and settings where points have little history of their own.
