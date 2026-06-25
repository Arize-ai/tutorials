import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

export const weatherAnalysisTool = createTool({
  id: 'weather-analysis-tool',
  description: 'Analyzes weather data to provide insights and patterns.',
  inputSchema: z.object({
    weatherData: z.string().describe('Raw weather data to be analyzed'),
    location: z.string().describe('Location name for context')
  }),
  outputSchema: z.object({
    analysis: z.string().describe('Detailed weather analysis and forecast interpretation')
  }),
  // Delegate to the weatherAnalysisAgent via the Mastra instance so the LLM
  // call is traced as a child span of this tool in Arize AX.
  execute: async ({ weatherData, location }, { mastra }) => {
    if (!mastra) {
      throw new Error('Mastra instance is not available in the tool execution context');
    }

    const agent = mastra.getAgent('weatherAnalysisAgent');
    const result = await agent.generate(
      `Analyze the following weather data for ${location} and provide detailed insights:

${weatherData}

Please provide:
1. Current conditions analysis and what they mean for different activities
2. Weather pattern trends and forecast insights
3. Optimal timing windows throughout the day
4. Any weather risks or special considerations
5. Overall assessment of weather suitability for outdoor vs indoor activities`
    );

    return {
      analysis: result.text
    };
  }
});
