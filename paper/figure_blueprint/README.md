# Fig. 1 blueprint: CricketMind architecture (for the Canva redraw)

**Look at this first:** `architecture_blueprint.png` (or the `.pdf`).
It is the target layout. Copy its structure; the Canva version may look nicer
(icons, fonts), but every box and arrow below must be there, and correct.

## 1. Size: the one hard rule

The paper is **exactly 6 pages, full to the last line**. The new figure must
not be taller than the old `Clip.png` on the page.

- Make the canvas **wider than tall, ratio at least 1.6 : 1**
  (e.g. 3200 x 2000 px). The old image was 2000 x 1400 (1.43 : 1).
- Export **PNG, at least 2000 px wide** (or PDF).
- Save it as **`Clip.png`** in `paper/CricketMind_ICCIT/`, replacing the old
  one, so the paper picks it up with no LaTeX change.
- Text in the figure must stay readable when printed about 16 cm wide:
  do not go smaller than the blueprint's text.

## 2. Four horizontal bands (name on the left, as in the example you liked)

| Band (top to bottom) | Tint | Strong colour for borders |
|---|---|---|
| Input & Perception | very light blue | blue `#2F6FC0` |
| Grounding & Analytics | very light gold | gold `#9A7B1F` |
| Agentic Core | very light green | green `#2E7D4F` (agents), purple `#5B3A9B` (router) |
| Verification & Output | very light purple | purple `#5B3A9B` |

Diamonds (decisions): yellow `#C99A06` border, light yellow fill.
Correction loops: **red `#D23B3B` dashed**. Exits: grey `#6B6B6B`.

## 3. Boxes: use exactly these labels

**Input & Perception**
- `User query` (white box, far left)
- ◆ `Clip?`
- `N1 Video input` / `15 frames`
- `N2 Vision encoder` / `EfficientNetV2-S`
- `N4 Temporal transformer`
- `N5 Classifier` / `15 shot classes`
- ◆ `Conf. check`
- `stop: low confidence` (grey exit)
- `N3 Text encoder`, **below User query**

**Grounding & Analytics**
- `N6 Shot DNA` / `shot + posture + filters`
- `N7a Stats retrieval`, with a small cylinder above it: `ball-by-ball DB`
- `N7b Context retrieval`, with a small cylinder below it: `historical DB`
- `N8 Stats engine` and `N9 What-if simulator`, inside one **dashed box**,
  with the side note `sanctioned engines: the only source of numbers`

**Agentic Core**
- `N10 Supervisor router` / `propose: keyword or LLM` / `rules: evidence, at most 4`
- The 12 agents in a **2 x 6 grid inside one dashed green box**:
  N11a Franchises, N11b Coaches, N11c Players, N11d Scouts, N11e Fans,
  N11f Stadiums, N11g Sponsors, N11h Broadcasters, N11i Governing bodies,
  N11j Academies, N11k Medical teams, N11l Player agents
- Note under the grid: `12 stakeholder agents; the activated ones run in parallel`

**Verification & Output**
- ◆ `Faith check`
- `N12 Synthesis` / `verified findings only`
- `N13 Final report` / `+ status`
- `withhold unverified` (grey)

## 4. Arrows: every one, with its label

| From | To | Label | Style |
|---|---|---|---|
| User query | Clip? | | solid |
| **User query** | **N3 Text encoder** | `every query` | solid |
| Clip? | N1 | `clip` | solid |
| N1 → N2 → N4 → N5 → Conf. check | | | solid chain |
| Conf. check | N1 (over the top) | `retry, at most 2: re-sample frames` | **red dashed** |
| Conf. check | stop: low confidence | `exhausted` | solid |
| Conf. check | N6 | `accept` | solid |
| Clip? | N6 (straight down) | `no clip` | solid |
| N3 | N6 | `filters` | solid, grey |
| N6 | N7a and N7b (one arrow that splits) | | solid |
| ball-by-ball DB → N7a, historical DB → N7b | | | solid, short |
| N7a → N8, N7b → N9 | | | solid |
| engines dashed box | N10 router | | solid |
| N10 router | agents dashed box | `activate` | solid |
| agents dashed box | Faith check | `findings + citations` | **green dashed** |
| **Faith check** | **agents dashed box** | `regenerate, at most 2: only the failed agents` | **red dashed** |
| Faith check | N12 | `faithful` | solid |
| Faith check | withhold unverified → N12 | `exhausted` | solid |
| N12 | N13 | | solid |

## 5. What this fixes compared with the old Clip.png

1. **N3 is fed on every path** (from User query), not only when there is no clip.
2. **Regenerate returns to the agents**, not to the N10 router.
3. **Agents are a parallel grid in one box**, not a chain of arrows N11a → N11b → ...
4. **Spelling:** "Stats", not "States"; "faithful", not "failthful".

## 6. Icons (optional)

Small icons are fine **if they do not make the figure taller**. Use only icons
you are allowed to use (Canva's own elements, or ones you credit); do not copy
icons from another paper's figure.

---
*The blueprint is drawn in TikZ: `paper/CricketMind_ICCIT/fig_architecture.tex`.
To regenerate the picture: `tectonic architecture_blueprint.tex` in this folder.*
