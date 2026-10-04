"""Investigation recommendations. Never recommends automatic bans or punishment."""


def recommendations(level: str, ind: dict | None = None) -> list[str]:
    ind = ind or {}
    base = {
        "HIGH": ["Review the posting timeline and interval distribution manually",
                 "Inspect repeated or near-duplicate content and shared URLs",
                 "Review the interaction network (who replies/reposts, reciprocity)",
                 "Check the account-creation timeline against its activity volume",
                 "Compare with known automated patterns (feeds, schedulers, promo/spam templates)"],
        "MEDIUM": ["Collect more history: this score is inconclusive", "Check whether the account is an organisation, news desk or social-media manager",
                   "Look for scheduling tools or cross-posting before drawing conclusions"],
        "LOW": ["No action needed based on this analysis", "Re-assess only if new evidence appears"],
    }[level][:]
    if ind.get("exact_duplicate_ratio", 0) > 0.3 or ind.get("template_ratio", 0) > 0.4:
        base.append("Content is highly templated — verify whether this is legitimate scheduled/brand messaging")
    if ind.get("circadian_concentration", 1) < 0.2:
        base.append("Activity spreads across all hours of the day — check for multi-operator or timezone effects")
    base.append("Do not suspend or label the account on the basis of this score alone; a human reviewer should decide")
    return base
