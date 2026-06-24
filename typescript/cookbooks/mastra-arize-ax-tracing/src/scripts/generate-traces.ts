/**
 * Sends a diverse set of prompts to the WeatherOrchestratorAgent to generate
 * traces in Arize AX. Each prompt is chosen to exercise a different tool path:
 *
 *   - weather-only            → weatherTool
 *   - weather+planning        → weatherTool → activityPlanningTool
 *   - weather+analysis+plan   → weatherTool → weatherAnalysisTool → activityPlanningTool
 *
 * Run with:  npm run generate-traces
 * Requires the same env vars as the agent: ARIZE_SPACE_ID, ARIZE_API_KEY, OPENAI_API_KEY.
 */
import { fileURLToPath } from 'node:url';

// The Mastra config points its storage at `file:../mastra.db`, resolved relative
// to the process working directory. Run from `src/` so that local db lands in
// the cookbook root (where *.db is gitignored) instead of the parent directory.
process.chdir(fileURLToPath(new URL('..', import.meta.url)));

const { mastra } = await import('../mastra/index');

interface Example {
  category: string;
  prompt: string;
}

const examples: Example[] = [
  // Simple weather lookups — the orchestrator should call weatherTool and stop.
  { category: 'weather-only', prompt: "What's the weather in Tokyo right now?" },
  { category: 'weather-only', prompt: 'Is it raining in Seattle at the moment?' },
  { category: 'weather-only', prompt: 'How hot does it feel in Dubai today?' },

  // Activity requests — weatherTool then activityPlanningTool.
  { category: 'weather+planning', prompt: 'What should I do in Barcelona today?' },
  { category: 'weather+planning', prompt: 'Give me some activity ideas for a day out in Amsterdam.' },
  { category: 'weather+planning', prompt: "I'm spending the day in Cape Town — what's good to do given the weather?" },

  // Deep dives — the full weatherTool → weatherAnalysisTool → activityPlanningTool chain.
  { category: 'weather+analysis+planning', prompt: 'Give me a detailed weather analysis and a full activity plan for London today.' },
  { category: 'weather+analysis+planning', prompt: 'Analyze the conditions in Reykjavik and plan my whole day around them.' },
  { category: 'weather+analysis+planning', prompt: 'I want an in-depth look at the weather in Singapore plus a complete itinerary for today.' },

  // Varied phrasing / edge cases for span diversity.
  { category: 'vague-intent', prompt: 'I might head outside in Chicago later... or maybe not. What do you reckon?' },
  { category: 'multi-location', prompt: 'Is it better weather for a long walk in Oslo or in Lisbon right now, and what should I do there?' },
  { category: 'activity-constrained', prompt: 'Plan an indoor-friendly day in Mumbai in case the weather is rough.' },
];

async function main(): Promise<void> {
  const agent = mastra.getAgent('weatherOrchestratorAgent');

  console.log(`Sending ${examples.length} prompts to WeatherOrchestratorAgent…\n`);

  let ok = 0;
  for (const [index, example] of examples.entries()) {
    const label = `[${index + 1}/${examples.length}] (${example.category})`;
    console.log(`${label} → ${example.prompt}`);
    try {
      const result = await agent.generate(example.prompt);
      const text = (result.text ?? '').replace(/\s+/g, ' ').trim();
      console.log(`   ✓ ${text.slice(0, 140)}${text.length > 140 ? '…' : ''}\n`);
      ok += 1;
    } catch (error) {
      console.error(`   ✗ failed: ${(error as Error).message}\n`);
    }
  }

  console.log(`Completed ${ok}/${examples.length} prompts. Flushing traces to Arize AX…`);
  // Cleanly shut down so the tracing exporter flushes all buffered spans before exit.
  await mastra.shutdown();
  console.log('Done. The traces should appear in your Arize AX space within a few seconds.');
}

try {
  await main();
  process.exit(0);
} catch (error) {
  console.error(error);
  process.exit(1);
}
