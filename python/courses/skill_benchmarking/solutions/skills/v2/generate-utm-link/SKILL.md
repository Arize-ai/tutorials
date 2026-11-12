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
- **Campaigns:** use a `slug` from [campaigns.csv](campaigns.csv) in this skill's folder. See [Campaigns](#campaigns).

## Campaigns

`campaigns.csv`, in this skill's folder, is a copy of the marketing team's
campaign list, the source of truth for campaign names.

- Read the file and match the request to one campaign, using its `name` and `description`.
- Copy the `slug` exactly. Never add a year, shorten it, or build a new name from the page or the date.
- Only use campaigns whose `status` is `active`.
- If no active campaign matches, don't make one up. Tell the user and ask which campaign to use.

## Steps

1. Get the destination URL.
1. Read `campaigns.csv` from this skill's folder and pick the campaign.
1. Pick the source and medium.
1. Add the parameters to the URL.
1. Return the link.

## Example

"Share the pricing page on LinkedIn for the spring launch" →
`https://brindlewick.example.com/pricing?utm_source=linkedin&utm_medium=social&utm_campaign=spring-launch`
