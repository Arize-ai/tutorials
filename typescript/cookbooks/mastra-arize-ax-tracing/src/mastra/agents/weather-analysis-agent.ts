import { openai } from '@ai-sdk/openai';
import { Agent } from '@mastra/core/agent';

// LLM-backed analysis step. Invoked by weatherAnalysisTool via the Mastra
// instance so the call is captured as a traced child span in Arize AX.
export const weatherAnalysisAgent = new Agent({
  id: 'weatherAnalysisAgent',
  name: 'WeatherAnalysisAgent',
  instructions: `
      You are a meteorological analyst. Given raw weather data for a location,
      provide a concise, specific analysis that covers:

      - Notable current conditions and what they mean in practical terms
      - Comfort and safety considerations (heat, cold, wind, precipitation, UV)
      - How the conditions are likely to affect outdoor plans for the day

      Respond with the analysis only. Do not ask follow-up questions.
`,
  model: openai('gpt-4.1-mini'),
});
