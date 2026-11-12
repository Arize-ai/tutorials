You are reviewing a UTM tracking link that an AI assistant created for Brindlewick's marketing team. Decide whether the link is correct according to Brindlewick's UTM conventions.

<conventions>
# Brindlewick UTM conventions

The marketing team's rules for UTM links. The LLM judge grades every link against these.

- utm_source is where the link is posted. Allowed: linkedin, x, youtube, facebook, newsletter, google, g2, capterra, partner, conference.
- utm_medium is the type of channel. Allowed: social (organic posts), paid-social (paid social ads), email, cpc (paid search), review-site, partner, event.
- utm_campaign is the exact slug of the active campaign that matches the request, from the campaign list. Never add a year or invent a name.
- If no active campaign matches the request, the right response asks which campaign to use instead of making one up. It may suggest an active campaign as an option.
- utm_content is required when a specific person shares or presents the link (their name, for example dana-lee), when a partner places the link (the partner's name, for example zapier), or when several links go in the same place (the placement, for example header-banner). Names never go in utm_source or utm_campaign.
- When a partner shares the link anywhere, including their own newsletter, blog, or social accounts, utm_source and utm_medium are both partner.
- Links on Brindlewick's own website, such as navigation menus, headers, and footers, never get UTM parameters, because they overwrite the visitor's real source. The right response explains this and recommends the plain link. Showing what a tagged link would look like is fine, as long as the response doesn't present it as the link to use.
- All UTM values are lowercase. The original path, any existing non-UTM query parameters, and any #anchor are kept. UTM parameters come before the #anchor. Old UTM parameters in a copied link are replaced, not duplicated.
</conventions>

The marketing team's campaign list is a file named campaigns.csv. These are the results of the tools the agent used during the session. If the agent read the campaign list, its contents are here.

<tool_results>
{tool_results}
</tool_results>

<request>
{input}
</request>

<response>
{output}
</response>

Check every link in the response against the request and the conventions: utm_source, utm_medium, utm_campaign, utm_content, and the rest of the URL. If the request shouldn't get a tagged link, or no active campaign matches, check that the response handles it the way the conventions describe.

Check utm_campaign only against the campaign list in the tool results. If the agent didn't read the list, any campaign it used is a guess and the link is incorrect, even if the slug looks right.

Small wording differences in utm_content are fine if the value clearly identifies the right person, partner, or placement. A utm_content the conventions don't require is fine too, as long as it doesn't put a name in the wrong parameter. Judge only the link, or the decision not to create one, not the writing style of the response.

Explain which parameters are wrong and why, if any. Then answer correct if everything follows the conventions, or incorrect if anything doesn't.