import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, "/opt/airflow")

from src.config import get_settings
from src.services.feishu.client import FeishuClient

from .common import get_cached_services

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DigestPaper:
    arxiv_id: str
    title: str
    authors: List[str]
    categories: List[str]
    published_date: datetime
    score: float
    reasons: List[str]
    abstract_excerpt: str


_CATEGORY_BONUS = {
    "cs.AI": 2.0,
    "cs.LG": 1.5,
    "cs.CL": 1.0,
    "cs.CV": 0.8,
    "stat.ML": 0.8,
}

_TITLE_KEYWORDS: Sequence[Tuple[str, float, str]] = (
    ("agent", 2.2, "Agentic workflow / tools"),
    ("reasoning", 2.0, "Reasoning"),
    ("benchmark", 1.8, "Benchmark / evaluation"),
    ("survey", 1.7, "Survey / review"),
    ("dataset", 1.5, "Dataset / benchmark resource"),
    ("retrieval", 1.4, "Retrieval / RAG"),
    ("multimodal", 1.4, "Multimodal"),
    ("planning", 1.4, "Planning"),
    ("alignment", 1.3, "Alignment / safety"),
    ("efficient", 1.1, "Efficiency"),
    ("scalable", 1.1, "Scalability"),
    ("robust", 1.0, "Robustness"),
    ("transformer", 1.0, "Transformer-related"),
)

_ABSTRACT_KEYWORDS: Sequence[Tuple[str, float, str]] = (
    ("state of the art", 1.6, "SOTA claim"),
    ("we propose", 1.4, "Proposed method"),
    ("we introduce", 1.4, "Introduces a new method"),
    ("open-source", 1.2, "Open-source artifact"),
    ("large language model", 1.8, "LLM-related"),
    ("retrieval augmented generation", 1.6, "RAG-related"),
    ("tool use", 1.4, "Tool use"),
    ("self-supervised", 1.2, "Self-supervised learning"),
    ("foundation model", 1.5, "Foundation model"),
    ("interpretability", 1.2, "Interpretability"),
    ("generalization", 1.0, "Generalization"),
)

_DIGEST_STATE_PATH = Path("data/paper_digests/digest_state.json")
_DISPLAY_TZ = timezone(timedelta(hours=8))


def _normalize_datetime(value: Optional[datetime]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_datetime_readable(value: datetime) -> str:
    local_dt = _normalize_datetime(value).astimezone(_DISPLAY_TZ)
    return local_dt.strftime("%Y-%m-%d %H:%M") + " (UTC+8)"


def _extract_excerpt(abstract: str, max_sentences: int = 2, max_chars: int = 260) -> str:
    text = " ".join((abstract or "").split())
    if not text:
        return ""

    sentence_chunks = []
    for chunk in text.replace("?", ".").replace("!", ".").split("."):
        clean = chunk.strip()
        if clean:
            sentence_chunks.append(clean)

    excerpt = ". ".join(sentence_chunks[:max_sentences]).strip()
    if excerpt and not excerpt.endswith("."):
        excerpt += "."

    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 1].rstrip() + "…"

    return excerpt


def _base_arxiv_id(arxiv_id: str) -> str:
    match = re.match(r"^(.*)v\d+$", arxiv_id)
    return match.group(1) if match else arxiv_id


def _load_digest_state() -> dict:
    if not _DIGEST_STATE_PATH.exists():
        return {"sent": {}}

    try:
        raw_state = json.loads(_DIGEST_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Digest state file is unreadable; starting from a clean state")
        return {"sent": {}}

    sent = raw_state.get("sent", {})
    if not isinstance(sent, dict):
        sent = {}

    return {"sent": sent}


def _save_digest_state(state: dict) -> None:
    _DIGEST_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DIGEST_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _purge_old_state(state: dict, retention_days: int, reference_time: datetime) -> dict:
    cutoff = reference_time - timedelta(days=retention_days)
    sent = state.get("sent", {})
    if not isinstance(sent, dict):
        return {"sent": {}}

    cleaned_sent = {}
    for arxiv_id, sent_at_text in sent.items():
        try:
            sent_at = datetime.fromisoformat(sent_at_text)
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            else:
                sent_at = sent_at.astimezone(timezone.utc)
            if sent_at >= cutoff:
                cleaned_sent[arxiv_id] = sent_at.isoformat()
        except Exception:
            continue

    return {"sent": cleaned_sent}


def _score_paper(paper, window_end: datetime) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    categories = [category for category in (paper.categories or []) if category]
    combined_text = f"{title} {abstract}"

    hours_old = max(0.0, (window_end - _normalize_datetime(getattr(paper, "published_date", None))).total_seconds() / 3600.0)
    recency_bonus = max(0.0, 3.0 - (hours_old / 24.0))
    if recency_bonus > 0:
        score += recency_bonus
        reasons.append(f"Recent ({hours_old:.1f}h old)")

    for category in categories:
        category_bonus = _CATEGORY_BONUS.get(category, 0.3 if category.startswith("cs.") else 0.1)
        if category_bonus > 0:
            score += category_bonus
            reasons.append(f"Category: {category}")

    for keyword, bonus, label in _TITLE_KEYWORDS:
        if keyword in title:
            score += bonus
            reasons.append(label)

    for keyword, bonus, label in _ABSTRACT_KEYWORDS:
        if keyword in combined_text:
            score += bonus
            reasons.append(label)

    if "benchmark" in combined_text and "evaluation" in combined_text:
        score += 0.8
        reasons.append("Benchmark + evaluation emphasis")

    if len((paper.abstract or "").split()) > 180:
        score += 0.4
        reasons.append("Substantial abstract")

    deduped_reasons = list(dict.fromkeys(reasons))[:4]
    return score, deduped_reasons


def _is_recent_duplicate(arxiv_id: str, sent_state: dict, window_end: datetime, suppression_days: int) -> bool:
    base_id = _base_arxiv_id(arxiv_id)
    sent_at_text = sent_state.get("sent", {}).get(base_id)
    if not sent_at_text:
        return False

    try:
        sent_at = datetime.fromisoformat(sent_at_text)
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        else:
            sent_at = sent_at.astimezone(timezone.utc)
    except Exception:
        return False

    return sent_at >= (window_end - timedelta(days=suppression_days))


def _format_authors(authors: Iterable[str], max_authors: int = 3) -> str:
    author_list = [author for author in authors if author]
    if not author_list:
        return "Unknown authors"
    if len(author_list) <= max_authors:
        return ", ".join(author_list)
    return ", ".join(author_list[:max_authors]) + " et al."


def _render_digest_markdown(window_start: datetime, window_end: datetime, papers: List[DigestPaper]) -> str:
    date_label = window_end.strftime("%Y-%m-%d")
    lines = [
        f"# arXiv 论文日报 · {date_label}",
        "",
        f"- 时间窗口: {_format_datetime_readable(window_start)} -> {_format_datetime_readable(window_end)}",
        f"- 入选论文: {len(papers)}",
        "",
    ]

    if not papers:
        lines.extend(
            [
                "## 今日无推荐论文",
                "",
                "时间窗口内没有检出足够新的论文，或者当前筛选阈值过高。",
            ]
        )
        return "\n".join(lines)

    lines.append("## 今日精选")
    lines.append("")

    for idx, paper in enumerate(papers, 1):
        lines.extend(
            [
                f"### {idx}. {paper.title}",
                f"- arXiv: https://arxiv.org/abs/{paper.arxiv_id}",
                f"- 作者: {_format_authors(paper.authors)}",
                f"- 分类: {', '.join(paper.categories) if paper.categories else 'unknown'}",
                f"- 发布时间: {_format_datetime_readable(paper.published_date)}",
                f"- 推荐原因: {', '.join(paper.reasons) if paper.reasons else '匹配了时间窗口和基础相关性规则'}",
            ]
        )
        if paper.abstract_excerpt:
            lines.append(f"- 摘要摘录: {paper.abstract_excerpt}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _trim_text(text: str, max_chars: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def _build_feishu_interactive_card(digest_result: dict) -> dict:
    papers = digest_result.get("selected_papers", [])
    window_start_text = _format_datetime_readable(datetime.fromisoformat(digest_result["window_start"]))
    window_end_text = _format_datetime_readable(datetime.fromisoformat(digest_result["window_end"]))
    date_label = _normalize_datetime(datetime.fromisoformat(digest_result["window_end"])).astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d")

    elements: List[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"**时间窗口**: {window_start_text} -> {window_end_text}\n"
                f"**入选论文**: {len(papers)}"
            ),
        },
        {"tag": "hr"},
    ]

    if not papers:
        elements.append(
            {
                "tag": "markdown",
                "content": "今天没有满足阈值与去重规则的论文。",
            }
        )
    else:
        for idx, paper in enumerate(papers, 1):
            title = _trim_text(str(paper.get("title") or "Untitled"), 180)
            arxiv_id = str(paper.get("arxiv_id") or "")
            authors = _trim_text(str(paper.get("authors") or "Unknown authors"), 120)
            categories = _trim_text(str(paper.get("categories") or "unknown"), 80)
            published_at = str(paper.get("published_at") or "")
            reasons = _trim_text(str(paper.get("reasons") or ""), 140)
            excerpt = _trim_text(str(paper.get("abstract_excerpt") or ""), 180)

            lines = [
                f"**{idx}. {title}**",
                f"[查看论文](https://arxiv.org/abs/{arxiv_id})",
                f"作者: {authors}",
                f"分类: {categories}",
                f"发布时间: {published_at}",
            ]
            if reasons:
                lines.append(f"推荐原因: {reasons}")
            if excerpt:
                lines.append(f"摘要: {excerpt}")

            elements.append({"tag": "markdown", "content": "\n".join(lines)})
            if idx != len(papers):
                elements.append({"tag": "hr"})

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"arXiv 论文日报 · {date_label}",
            }
        },
        "elements": elements,
    }


def _split_message(message: str, max_length: int = 3900) -> List[str]:
    if len(message) <= max_length:
        return [message]

    chunks: List[str] = []
    current = []
    current_length = 0

    for paragraph in message.split("\n\n"):
        paragraph_text = paragraph.strip()
        if not paragraph_text:
            continue
        paragraph_length = len(paragraph_text) + 2
        if current and current_length + paragraph_length > max_length:
            chunks.append("\n\n".join(current).strip())
            current = [paragraph_text]
            current_length = len(paragraph_text)
        else:
            current.append(paragraph_text)
            current_length += paragraph_length

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


async def _send_telegram_digest(bot_token: str, chat_id: str, message: str) -> None:
    try:
        from telegram import Bot
    except Exception as e:
        raise RuntimeError(
            "python-telegram-bot is not installed in Airflow runtime. "
            "Install dependencies and rebuild airflow image."
        ) from e

    bot = Bot(token=bot_token)
    for chunk in _split_message(message):
        await bot.send_message(chat_id=chat_id, text=chunk, disable_web_page_preview=True)


async def _send_feishu_digest(
    app_id: str,
    app_secret: str,
    receive_id: str,
    receive_id_type: str,
    digest_result: dict,
    message_fallback: str,
) -> None:
    client = FeishuClient(app_id=app_id, app_secret=app_secret)
    card = _build_feishu_interactive_card(digest_result)

    try:
        await client.send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type="interactive",
            content=card,
        )
    except Exception:
        # Fallback to plain text when interactive cards are restricted by tenant policy.
        for chunk in _split_message(message_fallback, max_length=3800):
            await client.send_message(
                receive_id=receive_id,
                receive_id_type=receive_id_type,
                msg_type="text",
                content={"text": chunk},
            )


def generate_daily_paper_digest(**context):
    """Generate a daily paper digest from recently stored arXiv papers."""

    logger.info("Generating daily paper digest")

    settings = get_settings()
    digest_settings = settings.digest

    if not digest_settings.enabled:
        logger.info("Daily digest is disabled by configuration")
        return {
            "status": "disabled",
            "message": "DIGEST__ENABLED=false",
            "papers_scanned": 0,
            "papers_selected": 0,
        }

    execution_date = _normalize_datetime(context.get("execution_date") or context.get("logical_date"))
    window_end = execution_date
    window_start = window_end - timedelta(days=digest_settings.lookback_days)

    _arxiv_client, _pdf_parser, database, _metadata_fetcher, _opensearch_client = get_cached_services()

    with database.get_session() as session:
        from sqlalchemy import desc

        from src.models.paper import Paper

        papers = (
            session.query(Paper)
            .filter(Paper.created_at >= window_start, Paper.created_at < window_end)
            .order_by(desc(Paper.published_date), desc(Paper.created_at))
            .all()
        )

    logger.info("Found %s recently stored papers for digest window", len(papers))

    digest_state = _purge_old_state(_load_digest_state(), digest_settings.duplicate_suppression_days + 30, window_end)

    scored_papers: List[DigestPaper] = []
    for paper in papers:
        if _is_recent_duplicate(paper.arxiv_id, digest_state, window_end, digest_settings.duplicate_suppression_days):
            logger.info("Skipping recently published paper: %s", paper.arxiv_id)
            continue

        score, reasons = _score_paper(paper, window_end)
        if score < digest_settings.min_score:
            logger.info("Skipping low-score paper %s (score %.2f < %.2f)", paper.arxiv_id, score, digest_settings.min_score)
            continue

        scored_papers.append(
            DigestPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=list(paper.authors or []),
                categories=list(paper.categories or []),
                published_date=_normalize_datetime(paper.published_date),
                score=score,
                reasons=reasons,
                abstract_excerpt=_extract_excerpt(paper.abstract),
            )
        )

    scored_papers.sort(key=lambda item: (item.score, item.published_date), reverse=True)
    selected_papers = scored_papers[: digest_settings.max_papers]
    digest_markdown = _render_digest_markdown(window_start, window_end, selected_papers)

    output_dir = Path("data/paper_digests")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"paper_digest_{window_end.strftime('%Y%m%d')}.md"
    output_path.write_text(digest_markdown, encoding="utf-8")

    digest_result = {
        "status": "success",
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "papers_scanned": len(papers),
        "papers_selected": len(selected_papers),
        "min_score": digest_settings.min_score,
        "duplicate_suppression_days": digest_settings.duplicate_suppression_days,
        "selected_arxiv_ids": [paper.arxiv_id for paper in selected_papers],
        "selected_papers": [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": _format_authors(paper.authors),
                "categories": ", ".join(paper.categories) if paper.categories else "unknown",
                "published_at": _format_datetime_readable(paper.published_date),
                "reasons": ", ".join(paper.reasons) if paper.reasons else "",
                "abstract_excerpt": paper.abstract_excerpt,
            }
            for paper in selected_papers
        ],
        "output_path": str(output_path),
        "output_preview": digest_markdown[:800],
    }

    if not selected_papers:
        digest_result["status"] = "below_threshold"
        digest_result["message"] = "No papers met the minimum digest score after duplicate suppression"

    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="paper_digest", value=digest_result)

    logger.info("Daily paper digest generated at %s", output_path)
    return digest_result


def publish_daily_paper_digest(**context):
    """Publish the generated digest to Telegram/Feishu when configured."""

    logger.info("Publishing daily paper digest")

    ti = context.get("ti")
    digest_result = ti.xcom_pull(task_ids="generate_daily_paper_digest", key="paper_digest") if ti else None

    if not digest_result:
        logger.warning("No digest found in XCom; nothing to publish")
        return {"status": "skipped", "message": "No digest found"}

    output_path = Path(digest_result["output_path"])
    if not output_path.exists():
        logger.warning("Digest file not found at %s", output_path)
        return {"status": "skipped", "message": f"Digest file not found: {output_path}"}

    if digest_result.get("status") != "success" or digest_result.get("papers_selected", 0) == 0:
        logger.info("Digest did not meet publish criteria; skipping Telegram push")
        return {
            "status": "skipped",
            "message": digest_result.get("message", "No papers met publish criteria"),
            "output_path": str(output_path),
            "min_score": digest_result.get("min_score"),
        }

    digest_markdown = output_path.read_text(encoding="utf-8")

    settings = get_settings()
    delivery_results = []

    if settings.telegram.enabled and settings.telegram.bot_token and settings.telegram.chat_id:
        try:
            asyncio.run(_send_telegram_digest(settings.telegram.bot_token, settings.telegram.chat_id, digest_markdown))
            delivery_results.append({"channel": "telegram", "status": "ok", "target": settings.telegram.chat_id})
        except Exception as e:
            logger.error("Telegram digest delivery failed: %s", e)
            delivery_results.append({"channel": "telegram", "status": "failed", "error": str(e)})

    if (
        settings.feishu.enabled
        and settings.feishu.app_id
        and settings.feishu.app_secret
        and settings.feishu.default_receive_id
    ):
        try:
            asyncio.run(
                _send_feishu_digest(
                    app_id=settings.feishu.app_id,
                    app_secret=settings.feishu.app_secret,
                    receive_id=settings.feishu.default_receive_id,
                    receive_id_type=settings.feishu.default_receive_id_type,
                    digest_result=digest_result,
                    message_fallback=digest_markdown,
                )
            )
            delivery_results.append({
                "channel": "feishu",
                "status": "ok",
                "target": settings.feishu.default_receive_id,
                "receive_id_type": settings.feishu.default_receive_id_type,
            })
        except Exception as e:
            logger.error("Feishu digest delivery failed: %s", e)
            delivery_results.append({"channel": "feishu", "status": "failed", "error": str(e)})

    if not any(item.get("status") == "ok" for item in delivery_results):
        logger.info("No digest channel configured or all deliveries failed")
        return {
            "status": "generated_only",
            "message": "No enabled digest delivery channel succeeded",
            "output_path": str(output_path),
            "delivery_results": delivery_results,
        }

    publish_result = {
        "status": "published",
        "output_path": str(output_path),
        "papers_selected": digest_result.get("papers_selected", 0),
        "delivery_results": delivery_results,
    }

    if ti:
        ti.xcom_push(key="paper_digest_publish_result", value=publish_result)

    execution_time = _normalize_datetime(context.get("execution_date") or context.get("logical_date"))
    digest_state = _purge_old_state(_load_digest_state(), settings.digest.duplicate_suppression_days + 30, execution_time)
    published_at = execution_time.isoformat()
    for paper_id in digest_result.get("selected_arxiv_ids", []):
        digest_state.setdefault("sent", {})[_base_arxiv_id(paper_id)] = published_at

    _save_digest_state(digest_state)

    logger.info("Daily paper digest published successfully via channels: %s", [item["channel"] for item in delivery_results if item.get("status") == "ok"])
    return publish_result