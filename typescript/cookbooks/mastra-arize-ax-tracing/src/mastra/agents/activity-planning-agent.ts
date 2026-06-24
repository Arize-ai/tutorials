import { openai } from '@ai-sdk/openai';
import { Agent } from '@mastra/core/agent';

// LLM-backed planning step. Invoked by activityPlanningTool via the Mastra
// instance so the call is captured as a traced child span in Arize AX.
export const activityPlanningAgent = new Agent({
  id: 'activityPlanningAgent',
  name: 'ActivityPlanningAgent',
  instructions: `
      You create detailed, location-specific activity plans from weather data
      (and an optional weather analysis). Recommend concrete activities with
      specific venues, parks, trails, or locations, sensible timing, and the
      weather considerations behind each choice. Always include indoor
      alternatives and any special considerations (gear, warnings).

      Follow the exact output format requested in the message. Respond with the
      plan only.
`,
  model: openai('gpt-4o-mini'),
});
