# Related work notes

For each paper: what they did, and how CricketMind differs. Add a BibTeX entry to
`references.bib` the same day. **Verify every paper yourself before citing it.** The first
entries came from a quick search on 2026-10-16; the systematic search of Day 13
(2026-10-21) is logged at the end of this file, with the comparison table and the novelty
verdicts.

## Verifying numbers and claims from LLMs (the closest work; position against it)

**Proof-Carrying Numbers (Solatorio, 2025, arXiv:2509.06902; v1 only).** Numbers are emitted
as claim-bound tokens tied to structured claims; a verifier in the *renderer* checks each one
under a declared policy (exact, rounding, tolerance), and anything unchecked defaults to
unverified. Formal results only (soundness, fail-closed); no empirical evaluation.
*How we differ:* the slots-and-renderer idea (our I3) is theirs too: cite it, do not claim
it. We add the subject-keyed re-query (C2, C6), wording against uncertainty (C4), tone
(C7) and a menu of claim kinds (registry).

**Tool receipts / NabaOS (Basu, 2026, arXiv:2603.10060).** HMAC-signed receipts for every tool
call; each claim is classed by its source (tool output, inference, testimony, absence,
opinion) and checked against the receipts the model cannot forge. 1,800 scenarios; catches
94.2% of fabricated tool references and 87.6% of count misstatements. Runtime detection.
*How we differ:* our receipts are for reproducibility (a reader re-runs the query), not
forgery detection; we verify against engines with subject-keyed re-queries; and we check
wording and tone. Receipts are not a novelty claim.

**EG-VAR (Junyu Ren, 2026, arXiv:2607.12650; ICML 2026 TAIGR workshop).** "Evidence-Grounded
Verified Agentic Reasoning: A Path Toward Eliminating LLM Hallucination in Empirical
Inference via Tool-Attested Kernel Proofs." A Lean 4 kernel is the only thing that can mark
a claim Verified, and only when it descends from an attested tool call; otherwise the
answer is Abstain, with a replayable audit trail. TableBench subset 120/120.
*How we differ:* the abstract checks where a number came from, not who it is about, how sure
the words sound, or what judgment they carry.

**VeriFin (Bethel Hall, Sachi Shome, William Eiers, 2026, arXiv:2608.10213; v2 17 Oct 2026).**
"VeriFin: A Neurosymbolic Framework for Verifying LLM-Generated Financial Claims." Operands
grounded in XBRL facts, the calculation taken from linkbases or documented metric
definitions, checked with Z3; an acceptance control that accepted none of the incorrect
claims (600-question XBRLFiling benchmark; FinanceBench).
*How we differ:* VeriFin already binds numbers to *what* they are about (period, unit, line
item), so subject binding is only partly ours (see verdicts). No uncertainty wording, no
tone.

**RAGShield (Patil, 2026, arXiv:2604.00387; v2 4 Apr 2026).** Links each dollar amount or
percentage in government text to its entity, then checks it against a value registry built
from the corpus; 430 attacks, all detected.
*How we differ:* entity linking for single values in text; our subjects have several filters
and are re-queried against an engine.

**Thucy (Theologitis and Suciu, 2026, arXiv:2512.03278; AAAI 2026 LaMAS workshop).**
Multi-agent claim verification across relational databases; reports the exact SQL behind
each verdict (TabFact 94.3%).
*How we differ:* it verifies claims someone else made; we gate what our own agents say. Its
"SQL behind each verdict" is close to our receipts: cite it there.

**GSAR (Kamelhar, 2026, arXiv:2604.23366).** Typed groundedness (grounded, ungrounded,
contradicted, complementary) for claims in multi-agent reports; proceed, regenerate or
replan under a compute budget.
*How we differ:* close to the Faith Check's containment loop (fix, retry, withhold): cite it
there. It scores groundedness with LLM judges; our checks are deterministic.

**Closed-World Resolution (Iyer, 2026, arXiv:2609.19425).** A taxonomy of tool hallucination
and a resolver that checks each tool call against a registry of tools and its signature;
mainly a measurement study (322 hallucinations, ten models).
*How we differ:* the nearest relative of our registry, but for tool calls and at call time;
ours lists kinds of claims and refuses hollow ones at start-up.

**Collective hallucination in multi-agent LLMs (Jamshidi, 2026, arXiv:2606.07941).** Models how
hallucinations spread between agents; trust weighting and isolation cut the hallucination
rate from 0.118 to 0.072.
*How we differ:* we prevent rather than measure. Claims are atomic by construction, and a
gate blocks them before release.

**Numerical claim verification (fact-checking).** CLEF CheckThat! 2025/2026 numeric tasks;
Aarnes and Setty, "Towards Robust Numerical Claim Verification" (AACL-IJCNLP 2026 Findings,
arXiv:2610.00689): small changes to a number break LLM checkers (frontier models about 74%
on label-flipping perturbations). *Use:* evidence for the AI-checker baselines that an LLM
is a weak judge of numbers; this field checks claims in the wild, not our own agents.

## LLMs expressing uncertainty

**Lin, Hilton and Evans (2022, TMLR; arXiv:2205.14334).** Teaching models to state their own
confidence in words.
**Wang et al. (2024, arXiv:2410.06707).** Calibrating verbalised probabilities.
**Yona et al. (EMNLP 2024; arXiv:2405.16908)**, **MetaFaith (Liu et al., EMNLP 2025;
arXiv:2505.24858)** and **Faithful Uncertainty Tuning (arXiv:2510.12587).** "Faithful
uncertainty": hedging that matches the model's *own* confidence.
*How we differ:* all of these concern a model's own confidence. We check whether an agent's
words match an *external engine's* interval, one that includes estimation uncertainty.

**Cerda-Mardini, Chandar and Madathil (2026, arXiv:2607.03882), "Consistent but
Miscalibrated: Evaluating LLM Limitations for Risk Communication in Natural Language."**
Nine LLMs choose words for simulated (Beta-distributed) predictions with known probability
and uncertainty; they are consistent but miscalibrated, worse on uncertainty than on
likelihood. A measurement study, no gate. **The closest work to C4**: it shows the problem
exists for external predictions, which is the case for enforcing it.

**Kent (1964), "Words of estimative probability".** The source of the wording bands. Newer
human and model studies to set the bands on Day 21: "How Unlikely Is 'Unlikely'?"
(arXiv:2608.26327, 19 models); "An Evaluation of Estimative Uncertainty in LLMs"
(arXiv:2405.15185).
**Gkatzia, Lemon and Rieser (ACL 2016), "Natural Language Generation enhances human
decision-making with uncertain information."** Supports giving people the range in words.

## Judgments, tone and framing

**Thomson and Reiter (INLG 2020; arXiv:2011.03992), "A Gold Standard Methodology for
Evaluating Accuracy in Data-To-Text Systems."** Human annotators marked 418 errors in 21
basketball summaries from three neural systems (about 20 each): 184 numbers, 105 named
entities, 80 words, 19 context. Their own example of an incorrect word is "strong" in "The
Grizzlies had a strong first half" when they did not do well. Human evaluation only.
*Use:* direct evidence, in sports, that wrong numbers (our F1), wrong names (F2) and wrong
judgment words (F8) are all common. Numbers in N-009.

**Balani and Panda (2026, arXiv:2609.27756), "Reporting Under Pressure: Separating Factual and
Tonal Sycophancy in LLM Statistical Analysis."** Tone shifts with the framing of the request
far more than facts do. Measurement only.
**FIFO (arXiv:2505.05406)** measures framing in LLM news summaries; **"Bias in the Tails"
(arXiv:2604.19984)** finds evaluative wording in resume summaries varying with the name.
*How we differ:* all measure tone; none requires each tone word to be backed by a checked
comparison.

## Honest answers: abstention and false premises

**Wen et al. (TACL 2025, vol. 13, pp. 529–556; arXiv:2407.18418), "Know Your Limits: A Survey
of Abstention in Large Language Models."** **AbstentionBench (arXiv:2506.09038)** includes
false-premise questions. **"Two Axes of LLM Abstention" (arXiv:2607.08456)**: a model
challenged no false premise under a standard prompt. Premise verification against a
knowledge graph (arXiv:2504.06438); **MultiHoax** (arXiv:2506.00264).
*How we differ:* abstention is well studied. Ours is narrower: every question is proved,
explained or declined, tied to the claim gate, with a read-back, and the shares are
measured on a locked exam with paraphrases.

## Cricket, sports and multi-agent systems

**FanCric (Bhatnagar, 2024, arXiv:2410.01307).** LLM agents selecting fantasy cricket teams.
Already cited in the ICCIT paper.
**CricBench (arXiv:2512.21877; v1 Devraj et al., Dec 2025; v3 Agarwal et al., Apr 2026, under
review).** Text-to-SQL on cricket: 633 base questions (each also in Hindi, Punjabi and Telugu),
over SQLite databases built from Cricsheet (Test, ODI, T20I, IPL). Most generated SQL runs, few
queries return the right answer. Checked 2026-10-19: the data and code are promised "upon
acceptance", with no URL or licence yet, and some of its tables do not add up (difficulty
counts, totals), so cite the abstract's wording and re-check the published version.
*Use:* evidence that cricket numbers produced by LLMs need verification (N-007), and, once
released, an outside question set for the evaluation (ask the authors for early access).
**MMCricBench-3K (MRL workshop 2025; arXiv:2508.17334).** Visual QA on 1,463 synthetic cricket
scorecard images; models struggle, more so in Hindi.
**CricketBench (Hugging Face dataset, 2026).** Multiple-choice cricket reasoning: laws, match
situations, statistics recall, commentary, tactics. Not yet read beyond the dataset card.
**CricGPT (FIRE, published Feb 2024; DOI 10.1145/3632754.3632757, unverified).** A GPT-aided
cricket QA system over ESPNcricinfo data and SQLite, about 85% accuracy reported. The ACM
page refused automated reading: find the PDF and confirm title, authors and year.
**Sriraman (2025), "A Multi-Agentic AI System for Cricket Team Selection," MSc thesis,
National College of Ireland (norma.ncirl.ie/9918).** A data-analyst flow and a team-selector
flow; reports 100% task success. Prior cricket LLM-agent work (it replaces the earlier
"IPL chatbot thesis" lead, which the search did not find).
**SUMMIR (Kumar et al., 2026, arXiv:2604.04947).** Ranks LLM-generated insights from 7,900
sports articles (cricket, soccer, basketball, baseball); facts scored with FactScore and
SummaC. Measures, does not gate.
**SportQA (Xia et al., NAACL 2024)** and **SportsMetrics (Hu et al., ACL 2024).** Sports
understanding and numeric reasoning benchmarks for LLMs.
**GridMind (arXiv:2504.08747)**, NFL questions with agents and RAG; **COACH
(arXiv:2512.01853)**, badminton video agents; multi-agent soccer understanding
(arXiv:2505.03735). Sports multi-agent systems; none verifies its numbers.
**A Survey of Large Models in Sports (Xu et al., Findings of ACL 2026; arXiv:2608.14377).** 241
papers; names hallucination as a barrier to trust and points to retrieval and grounding as
future directions. Read only in part: check Section 4 before citing it for the gap.

## Methods the engines and the bridge rely on

- Efron and Morris (1975): shrinkage (Stein's estimator), the classic baseball example.
- Brier (1950): scoring probability forecasts.
- Cohen (1960); Landis and Koch (1977): agreement, κ, and the "substantial" threshold.
- Ratner et al. (2017), Snorkel: weak supervision.
- Hulin et al. (2014); Impellizzeri et al. (2021): the workload ratio, and its critique.
- Pedregosa et al. (2011), scikit-learn; Ke et al. (2017), LightGBM, the histogram trees
  scikit-learn's boosted trees follow: Day 54's ML baseline (D-062, D-067).
- James (1981), log5: the odds rule for one player against another (D-061).
- Tango, Lichtman and Dolphin (2007), The Book: batter against pitcher adds little beyond
  the two players' records (E-033). Verify the chapter before citing.
- The beta-binomial fitted by maximum likelihood (Griffiths, 1973), and the likelihood
  principle (Berger and Wolpert, 1988): why batting until out does not bias it (D-060).
- Wolpert (1992), stacked generalization; Breiman (1996), stacked regressions: a second
  model learns how to weigh the first ones, on data they were not fitted on. Day 157's
  stacked mix of N9 and the trees (D-069).

## To read (verify that each exists, then add notes)

- Learned models (D-063, D-064), from the search of 2026-10-26: Mendes-Neves et al. (Machine
  Learning, 2024), Large Event Models for football; EventGPT (arXiv 2512.17266) and ScoutGPT
  (arXiv 2603.15212), player-conditioned event transformers with counterfactual rollouts;
  Alcorn, (batter|pitcher)2vec (MIT Sloan); an LSTM for ball-by-ball cricket win probability
  (arXiv 2011.02122); TabPFN-2.5 (arXiv 2511.08667) and TabICL (ICML 2025, arXiv
  2502.05564); LettuceDetect (arXiv 2502.17125), HHEM-2.1 and MiniCheck, hallucination
  detectors; GLiNER (arXiv 2311.08526); Qwen3 Embedding (arXiv 2506.05176); SetFit (Tunstall
  et al., 2022) and ModernBERT (Warner et al., 2024); DSPy and GEPA (arXiv 2507.19457);
  "Feeding LLM annotations to BERT classifiers at your own risk" (arXiv 2504.15432) and
  candidate-label distillation (ACL 2025); Venn-Abers calibration (van der Laan and Alaa,
  ICML 2025) and conformal win probabilities (arXiv 2208.08598); temperature scaling (Guo et
  al., 2017) and deep ensembles (Lakshminarayanan et al., 2017); ViTPose and RTMPose against
  MediaPipe; DeBERTa-v3 (He et al., 2023) and NLI for zero-shot classification (Yin et al.,
  2019), for the word lists against meaning (D-065).

- Grinsztajn, Oyallon and Varoquaux (NeurIPS 2022, Datasets and Benchmarks), "Why do
  tree-based models still outperform deep learning on tabular data?": why D-062's ML
  baseline is gradient-boosted trees, not a deep network.

- Agentic retrieval (D-055): Singh et al. (2025), "Agentic Retrieval-Augmented Generation:
  A Survey" (arXiv 2501.09136); Self-RAG (Asai et al., ICLR 2024); corrective RAG; LangGraph's
  agentic RAG tutorial (grade, rewrite, retry); SPORTSQL (arXiv 2508.17157), sports
  questions answered by SQL; an entity-linking agent for question answering (arXiv
  2508.03865); clarifying ambiguous entities (arXiv 2504.09665). Found by search on
  2026-10-25; read before citing.

- Uncertainty pictures (D-054): Kay et al. (CHI 2016), "When(ish) is my bus?", quantile
  dotplots; Hullman et al. (PLOS ONE 2015) and Kale et al. (TVCG 2019), hypothetical
  outcome plots; Padilla, Kay and Hullman (2020), a review of uncertainty visualization;
  "In Dice We Trust" (CHI 2024), trust in forecast displays over time. Found by search on
  2026-10-25; read before citing.

- Cricket player profiling from text commentary (arXiv:2311.06818) and fine-grain annotation
  of cricket videos from commentary (arXiv:1511.07607): leads for the commentary bridge.
- MARCH (arXiv:2603.24579), MACE (arXiv:2604.17225): multi-agent self-checking against
  documents and tables.
- "Chart-Supported or Model-Supplied?" (arXiv:2607.25021): numbers a model supplies that
  the chart does not hold.
- Self-correction and verification loops: Reflexion; Self-Refine; CRITIC.
- Gabbett (2016), "The training-injury prevention paradox" (British Journal of Sports
  Medicine): the commonly quoted workload-ratio cut-offs (0.8 to 1.3; above 1.5). Check
  the figures before citing them for D-044.
- Tool-grounded and program-aided reasoning: PAL; Toolformer.
- Fine-grained factuality: FActScore; RARR.
- Multi-agent frameworks: AutoGen; CrewAI; LangGraph documentation.

---

## Day 13: the systematic search (2026-10-21)

### Method

Sixteen fixed queries, chosen before searching, one per question the paper must answer
(is it new to check numbers / uncertainty words / tone / claim kinds / honest answers, and
what exists for cricket and sports). Each was run through Claude's web-search tool limited
to one site: arXiv, the ACL Anthology, Semantic Scholar, or the open web in place of Google
Scholar (which cannot be searched that way). Six targeted look-ups then confirmed details of
known papers. Papers were kept when they check, measure or survey something CricketMind
claims; their abstract pages were read (Thomson and Reiter: Sections 1–4 in full).

**Two honest limits.** "Shown" is the number of results the tool returned, not the
database's total; and the verdicts below rest on abstracts. Both are fixed by the "you"
items at the end.

### Search log

| # | Date | Database | Words (exactly as run) | Shown | Kept |
|---|---|---|---|---|---|
| S1 | 2026-10-21 | arXiv | LLM numeric claim verification | 10 | PCN; RAGShield; VeriFin; Towards Robust Numerical Claim Verification; CheckThat! 2026 |
| S2 | 2026-10-21 | ACL Anthology | LLM numeric claim verification | 9 | Test-Time Scaling for Numerical Claim Verification; Claim Verification in the Age of LLMs |
| S3 | 2026-10-21 | Semantic Scholar | LLM numeric claim verification | 9 | Chart-Supported or Model-Supplied? |
| S4 | 2026-10-21 | arXiv | faithful uncertainty language model verbal hedging matches interval | 9 | Yona et al.; FUT; Consistent but Miscalibrated |
| S5 | 2026-10-21 | ACL Anthology | faithful uncertainty language model | 9 | Yona et al. (EMNLP 2024); MetaFaith (EMNLP 2025) |
| S6 | 2026-10-21 | arXiv | multi-agent sports analytics LLM | 9 | FanCric; GridMind; COACH; Survey of Large Models in Sports; soccer MAS |
| S7 | 2026-10-21 | arXiv | cricket question answering LLM | 9 | CricBench; MMCricBench-3K |
| S8 | 2026-10-21 | open web | cricket question answering large language model | 9 | CricGPT; CricketBench; MMCricBench-3K |
| S9 | 2026-10-21 | arXiv + ACL Anthology | evaluative language unsupported by data LLM statistics summaries misleading framing | 9 | Reporting Under Pressure; FIFO; Bias in the Tails |
| S10 | 2026-10-21 | arXiv + ACL Anthology | words of estimative probability language models perception verbal probability expressions | 9 | How Unlikely Is "Unlikely"?; Estimative Uncertainty in LLMs |
| S11 | 2026-10-21 | arXiv + ACL Anthology | abstention large language models survey know your limits | 10 | Wen et al. (TACL 2025); AbstentionBench |
| S12 | 2026-10-21 | arXiv | LLM agent typed claims schema registry checked at build time hallucinated tool capabilities | 9 | Closed-World Resolution; GSAR; Skills That Don't Exist |
| S13 | 2026-10-21 | arXiv + ACL Anthology | sports data-to-text LLM hallucination numbers wrong player faithfulness | 10 | SportsMetrics; SUMMIR; Input Matters |
| S14 | 2026-10-21 | arXiv | multi-agent LLM verification gate before release numeric claims engine re-query | 10 | MARCH; MACE; Thucy |
| S15 | 2026-10-21 | arXiv + ACL Anthology | question answering questionable assumptions false premise detection LLM | 9 | Two Axes of LLM Abstention; premise verification via KG; MultiHoax |
| S16 | 2026-10-21 | ACL Anthology | Thomson Reiter gold standard methodology evaluating accuracy data-to-text basketball error categories | 9 | Thomson and Reiter (INLG 2020); Accuracy shared task (INLG 2021) |
| T1–T6 | 2026-10-21 | targeted | EG-VAR and VeriFin by arXiv ID; SportQA and SportsMetrics; "CricGPT"; the NCI cricket thesis; Gkatzia 2016 | — | details confirmed (CricGPT only partly) |

### What the closest work checks

"—" means the abstract does not say it does.

| Work | Checks | Gate or measure | Who it is about | Words vs uncertainty | Tone words | Kinds of claim |
|---|---|---|---|---|---|---|
| PCN (2025) | each shown number links to a verified claim; verifier in the renderer | gate (fail-closed) | — | — | — | — |
| Tool receipts (2026) | claims vs signed tool receipts | runtime detection | — | — | — | — |
| EG-VAR (2026) | verified only if derived from an attested tool call | gate (else Abstain) | — | — | — | — |
| VeriFin (2026) | financial operands vs filed facts; calculation by Z3 | gate | partly (period, unit, line item) | — | — | authorised calculations |
| RAGShield (2026) | values vs a corpus-built value registry | detection | yes (entity linking) | — | — | — |
| Thucy (2026) | claims vs relational databases, SQL shown | verdicts | — | — | — | — |
| GSAR (2026) | groundedness of claims in multi-agent reports | runtime control | — | — | — | — |
| Closed-World Resolution (2026) | tool calls vs a tool registry | measurement + resolver | — | — | — | tools, at call time |
| Consistent but Miscalibrated (2026) | chosen words vs known probability and uncertainty | measure | — | **yes, measured** | — | — |
| Yona / MetaFaith / FUT | hedging vs the model's own confidence | measure / train | — | own confidence | — | — |
| Thomson and Reiter (2020) | human-marked errors in basketball summaries | measure (human) | yes (name errors) | — | **yes ("strong")** | — |
| Reporting Under Pressure (2026) | tonal vs factual divergence | measure | — | — | partly | — |
| **CricketMind v4** | **every claim, re-queried** | **gate** | **multi-filter subject** | **external interval** | **checked comparison** | **menu, at start-up** |

### Verdicts on the v4 document's novelty claims (§13)

| # | Claim in v4 | Verdict | Closest | What stays ours |
|---|---|---|---|---|
| 1 | Subject binding | **partly done** | VeriFin (period, unit, line item); RAGShield (entity linking); name errors in sports summaries (Thomson and Reiter) | re-querying each claim for a subject made of several filters at once (player, shot, phase, bowler type, …), plus the prose's names checked against it (C2, C6) |
| 2 | Uncertainty wording checked against an external interval | **still new as a gate** | Consistent but Miscalibrated measured the problem; faithful-uncertainty work uses the model's own confidence | enforcing the fit before release (C4) |
| 3 | Judgments as checked comparisons | **still new** | Thomson and Reiter ("strong" as an incorrect word, human-marked); tone measured in 2026 work | requiring each tone word, and each premise, to be backed by a comparison pointing the same way (C7, I7) |
| 4 | Registry of claim kinds, at build time | **still new in this form, minor** | Closed-World Resolution (tools, call time); VeriFin (authorised calculations) | a method component, not a headline contribution |
| 5 | Honest, measured coverage | **partly done** | abstention survey and benchmarks; EG-VAR abstains | the three-way outcome tied to the gate, the read-back, measured on paraphrases: an evaluation design, not a headline |
| — | Agents never type numbers (slots + renderer) | **already done by PCN** | PCN | not claimed; cite PCN |
| — | Receipts | **already done in other forms** | Tool receipts; Thucy's SQL per verdict | not claimed; cite both |
| — | Fix, retry, withhold loop | **already done in other forms** | GSAR; collective hallucination | not claimed; cite both |
| — | Same checks in two sports | a demonstration of generality, not a novelty claim | — | — |

**So the paper's headline contributions are 1 (narrowed), 2 and 3,** backed by the
evaluation. 4 and 5 are presented as parts of the method (D-026).

### Still to do (you)

- [ ] Read in full, in this order: PCN (how its tokens and policies work: compare with our
      slots); VeriFin (how operands are bound: compare with C2); Consistent but Miscalibrated
      (their descriptor scale: useful for C4's bands); Thomson and Reiter (their categories
      vs our F1–F8); EG-VAR (what "Abstain" covers). Change a verdict if the full text
      disagrees with the abstract.
- [ ] Re-run S1, S4, S7 and S9 on each site's own search page (arxiv.org/search,
      aclanthology.org, semanticscholar.org, scholar.google.com) and add the true hit counts.
- [ ] Find the CricGPT PDF and confirm its title, authors and year.
- [ ] Confirm the ICCIT paper's final title, authors and year after the decision.
- [ ] Repeat the whole search before submission: this area moves monthly.
