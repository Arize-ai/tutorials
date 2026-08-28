# 15-minute walkthrough outline

Use this outline to record the walkthrough embedded in the companion cookbook after publication.

| Time | Scene | Key point |
| --- | --- | --- |
| 0:00–1:30 | Show `images/` and three receipt types | Receipt extraction errors can still be costly. |
| 1:30–4:00 | Launch Gradio and run `orchard-market` | `gpt-5.4-mini` returns structured extraction and traces image context. |
| 4:00–6:00 | Open the AX project and inspect `receipt.extract` | Show receipt ID, scenario, model, output, and selectable image URL. |
| 6:00–9:00 | Create the span-level evaluator | Map receipt image, extraction output, and scenario; define `grounded`, `not_grounded`, and `needs_review`; choose `gpt-5.6-terra`. |
| 9:00–11:00 | Compare a clear and an ambiguous receipt | Show why the judge selects `grounded` or `needs_review` from the image evidence. |
| 11:00–12:30 | Run `faded-fern` or `blurred-bay` | Show why an ambiguous receipt should receive `needs_review`. |
| 12:30–14:00 | Run `python receipt_app.py --batch` and filter in AX | Compare evaluator behavior across standard, currency/tip, rotated, and degraded scenarios. |
| 14:00–15:00 | Review calibration and cost | Image cost varies with resolution/detail; calibrate the stronger judge against human labels before deciding a cheaper extractor is sufficient. |

Before recording, set `RECEIPT_IMAGE_BASE_URL` to an HTTPS fixture host and verify that AX renders the image field. Do not show real receipts, API keys, or customer data.
