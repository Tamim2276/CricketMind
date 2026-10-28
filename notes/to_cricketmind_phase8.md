# Phase 8 update: CricShot hands over its whole agent as the classifier node

From the CricShot side, 2026-10-29. Please update `docs/BUILD_PLAN.md`, Phase 8
(Days 112–128), to match.

## The decision

CricShot will not hand over one classifier. It hands over its **whole
shot-classification agent** (CricShot's Day 44, built after its Phase 4), and
CricketMind uses it as its classifier node:

```text
clip -> detect_and_crop -> quality_check -> fast_model
          confident -> answer
          unsure    -> expert models in parallel -> weighted vote -> explain -> answer
```

The agent is already a confidence gate: easy clips are answered by one fast
model, unsure ones go to several experts that see the clip differently
(motion, single frames, pose), and their weighted vote decides. Every node is
a plain function around a trained model; there is no LLM in it. The models,
the vote weights and the confidence threshold are fixed by CricShot on its
validation set before the handover.

## What the handover contains (Day 112)

- The agent's code, as a LangGraph subgraph CricketMind can add as one node
- The trained weights of every model in it, with checksums
- `classes.json`
- A model card for each model, and an **agent card**: which models, the
  threshold, the vote weights, and the exact input, step by step
- For the 20 parity clips: the frame numbers picked, the crops, each model's
  probabilities and the agent's final answer
- **Not TorchScript.** It is deprecated in torch 2.13, and some of the video
  models (VideoMAE, the pytorchvideo ones) may not convert. Weights plus model
  code, or `torch.export`, decided at export time.

## The contract

- **Input:** the raw clip. Frame picking, the batter crop and the quality
  check come inside the agent, so CricketMind cannot get them subtly wrong.
  Each model converts frames to what its weights expect (frame count, size,
  colours) internally.
- **Output:** the shot label, its confidence, the top 3, a status
  (`ok`, `low_confidence` or `cannot_classify`), which models were asked and
  what each said, and the frames that mattered.
- It runs unmodified on the Arc (XPU), CUDA and CPU.

## Day-by-day changes

| Day | Was | Becomes |
| --- | --- | --- |
| 112 | export one classifier as TorchScript | import the agent package; checksums match |
| 113 | device helper | unchanged; the agent must pass on both machines |
| 114–115 | rebuild frame picking and the striker crop | not rebuilt: they ship inside the agent. Only wrap the agent |
| 116 | parity of your frames and crops with CricShot's | parity of **the agent run inside CricketMind** with the package: same frame numbers, same crops, same probabilities and answer on all 20 clips |
| 117 | classifier adapter | a thin adapter that calls the agent subgraph; the stub stays for tests |
| 118 | confidence gate, retry up to 2 times, low-confidence exit | **no blind retry**: the same model on the same frames gives the same answer. The agent's own route is the retry (fast model, then experts). Its `low_confidence` status is the low-confidence exit; `cannot_classify` is new, for clips with no readable batter |
| 119 | video path into the graph and API | the agent as a subgraph node, end to end with a real clip |
| 120–125 | pose | unchanged. If the agent has a pose expert, share its keypoints with N2b instead of running pose twice; the posture measurements and their AUC stay CricketMind's own work |
| 127 | website shows shot and confidence | also which experts were asked and whether they agreed; still never accuracy |

## Rules that do not change

- **No CricShot accuracy numbers anywhere in CricketMind:** not in the
  papers, the website or the slides. They belong to the CricShot accuracy
  paper. CricketMind describes the agent's architecture (routing, the gate,
  the vote, the refusal) without measured accuracy.
- The parity test stays exact on frame numbers, not only a pixel tolerance.

## Timing

CricShot builds the agent after its Phase 4, about 3–4 weeks from now.
CricketMind's Phase 8 starts at its Day 112, so the agent will be ready
before it is needed. Until then the stub stands in, as planned.
