"""HTML renderers for the receipt expense inbox."""

from __future__ import annotations

from html import escape
from typing import Any

EXPENSE_INBOX_HEADER = (
    '<div class="section-title">Expense inbox</div>'
    '<div class="section-copy">Select a submitted receipt and create a structured expense record.</div>'
)
SUBMITTED_RECEIPT_LABEL = '<div class="field-label">Submitted receipt</div>'
STRUCTURED_RECORD_LABEL = '<div class="field-label">Structured expense record</div>'
INITIAL_EXPENSE_SUMMARY = (
    '<div class="expense-summary"><h3>Expense details</h3>'
    '<div class="section-copy">Process a receipt to create an expense record.</div></div>'
)


def render_stat(label: str, value: str) -> str:
    """Render one labeled value in the expense summary."""
    return f"<div><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"


def render_expense_summary(record: dict[str, Any]) -> str:
    """Render an extracted receipt as an expense summary card."""
    currency = str(record.get("currency") or "—")
    merchant = str(record.get("merchant") or "Merchant pending review")
    total = record.get("total")
    total_value = f"{currency} {total:,.2f}" if isinstance(total, (int, float)) else "Amount pending review"
    review_needed = bool(record.get("needs_review", False))
    review_class = "review-needed" if review_needed else "review-ok"
    review_text = "Review needed" if review_needed else "Ready for review"
    stats = "".join(
        (
            render_stat("Expense type", "Receipt"),
            render_stat("Line items", str(len(record.get("items") or []))),
            render_stat("Currency", currency),
        )
    )
    return (
        '<div class="expense-summary">'
        f"<h3>{escape(merchant)}</h3>"
        f'<div class="expense-total">{escape(total_value)}</div>'
        f'<div class="expense-meta">{stats}</div>'
        f'<span class="review-badge {review_class}">{review_text}</span>'
        "</div>"
    )


def render_queue_status(*, total_receipts: int, processed_count: int) -> str:
    """Render the pending and processed receipt counts."""
    pending_count = total_receipts - processed_count
    return f'<div class="queue-status">{pending_count} pending · {processed_count} processed</div>'


def render_processing_status() -> str:
    """Render successful processing and trace status."""
    return '<div class="trace-status">Trace created in AX. · Receipt moved to Processed.</div>'


def render_processing_error(error: Exception) -> str:
    """Render a safely escaped extraction error."""
    return f'<div class="trace-status">Processing failed: {escape(str(error))}</div>'
