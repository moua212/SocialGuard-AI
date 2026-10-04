# Limitations, fairness and responsible use

SocialGuard AI outputs a **probability of automated behavior**. It is a risk-assessment aid for human analysts, not an identity-verification or enforcement system.

| Limitation | Consequence |
|---|---|
| High posting frequency does not prove automation | Professional social-media managers, community moderators and power users can score high. |
| News, brand and customer-support accounts post often and regularly | Expect false positives; the `pro_human` archetype in the synthetic data exists to measure exactly this. |
| Scheduled content (Buffer, Hootsuite, RSS cross-posting) resembles bots | "Automated" ≠ "malicious". Many automated accounts are legitimate. |
| Hour-of-day features use UTC | Timezone is unknown; we add timezone-invariant features (circadian concentration, longest quiet period) but residual noise remains. |
| NLP features are lexical/embedding based and mostly language-agnostic, but the language detector is a tiny stop-word heuristic | Non-Latin scripts and low-resource languages are less reliable; sentence-transformers (multilingual) is recommended. |
| Platform-specific behavior | A model trained on one platform's norms will shift on another; retrain or re-calibrate per platform. |
| Synthetic training data | The bundled benchmark validates the pipeline. It cannot be used to claim real-world accuracy. Evaluate on labelled real data (with licence) before any operational use. |
| Label noise & dataset bias | Public bot datasets reflect old bot generations; modern LLM-driven bots mimic human text and timing. |
| Adversarial adaptation | Operators can randomize intervals and paraphrase text; the detector must be retrained and monitored. |
| Quick-scoring mode | `/api/predict` with only a few numbers uses a reduced model with no content evidence; its warning field says so. |

## Responsible-use rules built into the product
* Wording is always probabilistic ("High probability of automated behavior").
* Uncertain band: scores between the validated LOW and HIGH thresholds are labelled *inconclusive*.
* Recommendations never include automatic bans; every high-risk result ends with "a human reviewer should decide".
* The LLM layer only narrates supplied evidence, cannot alter scores, and falls back to a deterministic template.
* Thresholds are derived on out-of-fold validation probabilities (precision ≥ 90% for HIGH, recall ≥ 95% for the LOW cut-off) and recorded in the model metadata.
