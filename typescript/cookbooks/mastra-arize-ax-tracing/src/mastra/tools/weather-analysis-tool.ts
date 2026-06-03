import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

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
  execute: async ({ context }) => {
    const { weatherData, location } = context;

    const result = await generateText({
      model: openai('gpt-4o-mini'),
      prompt: `Analyze the following weather data for ${location} and provide detailed insights:

${weatherData}

Please provide:
1. Current conditions analysis and what they mean for different activities
2. Weather pattern trends and forecast insights
3. Optimal timing windows throughout the day
4. Any weather risks or special considerations
5. Overall assessment of weather suitability for outdoor vs indoor activities`
    });

    return {
      analysis: result.text
    };
  }
});
