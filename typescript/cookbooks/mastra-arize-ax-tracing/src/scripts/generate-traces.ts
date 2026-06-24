/**
 * Sends a diverse set of prompts to the WeatherOrchestratorAgent to generate
 * traces in Arize AX. The prompts live in ./example-prompts.json, grouped by
 * the tool path each one is meant to exercise:
 *
 *   - weather-only            → weatherTool
 *   - weather+planning        → weatherTool → activityPlanningTool
 *   - weather+analysis+plan   → weatherTool → weatherAnalysisTool → activityPlanningTool
 *   - multi-location / vague-intent / activity-constrained → varied phrasing
 *
 * Run with:  npm run generate-traces
 * Requires the same env vars as the agent: ARIZE_SPACE_ID, ARIZE_API_KEY, OPENAI_API_KEY.
 */
import { readFileSync } from 'node:fs';
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

// Prompts grouped by category in the JSON file; flattened to a run list here.
const promptsByCategory = JSON.parse(
  readFileSync(new URL('./example-prompts.json', import.meta.url), 'utf8'),
) as Record<string, string[]>;

const examples: Example[] = Object.entries(promptsByCategory).flatMap(
  ([category, prompts]) => prompts.map((prompt) => ({ category, prompt })),
);

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
