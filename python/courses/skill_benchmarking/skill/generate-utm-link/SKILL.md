---
name: generate-utm-link
description: Create UTM tracking links that follow Brindlewick's conventions. Use when someone asks for a UTM link, tracking link, or campaign link.
---

# Generate UTM link

Every link we share outside Brindlewick needs UTM parameters so we can see
which channels drive signups.

## Parameters

| Parameter    | Required | Meaning                                |
|--------------|----------|----------------------------------------|
| utm_source   | Yes      | Where the link is posted               |
| utm_medium   | Yes      | The type of channel                    |
| utm_campaign | Yes      | The campaign the link belongs to       |
| utm_content  | No       | Tells apart links in the same campaign |

## Allowed values

- **Sources:** linkedin, x, youtube, facebook, newsletter, google, g2, capterra, partner, conference
- **Mediums:** social, paid-social, email, cpc, review-site, partner, event
- **Campaigns:** the campaign name in lowercase with hyphens, for example `spring-launch`

## Steps

1. Get the destination URL.
1. Pick the source, medium, and campaign.
1. Add the parameters to the URL.
1. Return the link.

## Example

"Share the pricing page on LinkedIn for the spring launch" →
`https://brindlewick.example.com/pricing?utm_source=linkedin&utm_medium=social&utm_campaign=spring-launch`
