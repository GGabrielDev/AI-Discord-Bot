from urllib.parse import urlparse


LOW_SIGNAL_URL_TERMS = (
    "/search",
    "/tag/",
    "/tags/",
    "/category/",
    "/categories/",
    "/login",
    "/signin",
    "/signup",
    "/register",
    "/account",
    "/advert",
    "/ads",
)

HIGH_SIGNAL_HOSTS = {
    "github.com": 2.0,
    "docs.github.com": 2.0,
    "developer.mozilla.org": 2.0,
    "wikipedia.org": 1.5,
}

HIGH_SIGNAL_PATH_TERMS = ("docs", "reference", "manual", "guide", "api", "spec")


def _query_terms(query: str) -> set[str]:
    return {term for term in query.lower().split() if len(term) >= 4}


def score_search_result(result: dict, query: str, domain_counts: dict[str, int] | None = None) -> tuple[float, list[str]]:
    title = (result.get("title") or "").lower()
    snippet = (result.get("snippet") or "").lower()
    url = (result.get("url") or "").strip()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query_terms = _query_terms(query)
    reasons = []
    score = 0.0

    if parsed.scheme not in {"http", "https"} or not host:
        return -999.0, ["invalid-url"]

    matched_title_terms = sum(1 for term in query_terms if term in title)
    matched_snippet_terms = sum(1 for term in query_terms if term in snippet)

    if matched_title_terms:
        score += matched_title_terms * 1.2
        reasons.append(f"title-match:{matched_title_terms}")
    if matched_snippet_terms:
        score += matched_snippet_terms * 0.8
        reasons.append(f"snippet-match:{matched_snippet_terms}")

    if len(snippet) >= 140:
        score += 0.8
        reasons.append("rich-snippet")
    elif snippet:
        score += 0.2
    else:
        score -= 0.8
        reasons.append("missing-snippet")

    if host.endswith(".gov") or host.endswith(".edu"):
        score += 1.8
        reasons.append("trusted-domain")
    elif host.endswith(".org"):
        score += 0.8
        reasons.append("org-domain")

    for known_host, bonus in HIGH_SIGNAL_HOSTS.items():
        if host == known_host or host.endswith(f".{known_host}"):
            score += bonus
            reasons.append("known-source")
            break

    if any(term in path for term in HIGH_SIGNAL_PATH_TERMS):
        score += 0.6
        reasons.append("reference-path")

    if any(term in path for term in LOW_SIGNAL_URL_TERMS):
        score -= 2.0
        reasons.append("low-signal-path")

    if url.lower().endswith(".pdf"):
        score -= 0.4
        reasons.append("pdf-cost")

    if domain_counts is not None:
        repeats = domain_counts.get(host, 0)
        if repeats:
            penalty = min(1.5, repeats * 0.6)
            score -= penalty
            reasons.append("domain-saturation")

    return score, reasons


def prefilter_search_results(
    results: list[dict],
    query: str,
    max_results: int,
    min_score: float,
) -> tuple[list[dict], list[dict]]:
    scored_results = []
    domain_counts: dict[str, int] = {}

    for result in results:
        score, reasons = score_search_result(result, query, domain_counts)
        enriched = dict(result)
        enriched["prefilter_score"] = score
        enriched["prefilter_reasons"] = reasons
        scored_results.append(enriched)

        host = urlparse((result.get("url") or "").strip()).netloc.lower()
        if host:
            domain_counts[host] = domain_counts.get(host, 0) + 1

    scored_results.sort(key=lambda item: item.get("prefilter_score", float("-inf")), reverse=True)

    accepted = [item for item in scored_results if item.get("prefilter_score", -999.0) >= min_score][:max_results]
    if not accepted and scored_results:
        accepted = scored_results[:1]

    accepted_urls = {item.get("url") for item in accepted}
    rejected = [item for item in scored_results if item.get("url") not in accepted_urls]
    return accepted, rejected
