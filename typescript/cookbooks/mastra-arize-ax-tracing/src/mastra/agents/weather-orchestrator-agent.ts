import { openai } from '@ai-sdk/openai';
import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { weatherTool } from '../tools/weather-tool';
import { weatherAnalysisTool } from '../tools/weather-analysis-tool';
import { activityPlanningTool } from '../tools/activity-planning-tool';

export const weatherOrchestratorAgent = new Agent({
  id: 'weatherOrchestratorAgent',
  name: 'WeatherOrchestratorAgent',
  instructions: `
      You are the Weather Orchestrator Agent. You MUST use the available tools to provide weather information and recommendations.

      Available Tools (USE THESE TOOLS, NOT DIRECT LLM RESPONSES):
      🌤️ weatherTool: Fetches current weather data (temperature, humidity, wind, conditions)
      🔍 weatherAnalysisTool: Analyzes weather data for insights and patterns  
      📅 activityPlanningTool: Creates activity recommendations based on weather

      MANDATORY FLOW RULES:
      1. ALWAYS start with weatherTool to get current weather data
      2. For simple weather queries: Use weatherTool only
      3. For activity recommendations: Use weatherTool → activityPlanningTool
      4. For detailed analysis: Use weatherTool → weatherAnalysisTool → activityPlanningTool
      5. NEVER provide weather information without using weatherTool first
      6. NEVER provide activity recommendations without using activityPlanningTool
      7. NEVER provide analysis without using weatherAnalysisTool

      TOOL USAGE REQUIREMENTS:
      - You MUST call tools to get data, not generate responses directly
      - Pass data between tools as required (weather data to analysis/planning tools)
      - Always use the tool outputs to construct your final response
      - If a tool fails, explain the error and suggest alternatives

      Examples of REQUIRED tool usage:
      - "What's the weather?" → Call weatherTool, then respond with its output
      - "What should I do today?" → Call weatherTool, then activityPlanningTool, then respond
      - "Analyze the weather" → Call weatherTool, then weatherAnalysisTool, then respond

      Remember: You are a tool orchestrator, not a direct weather information provider.
`,
  model: openai('gpt-4o-mini'),
  tools: { 
    weatherTool,
    weatherAnalysisTool,
    activityPlanningTool
  },
  memory: new Memory({
    storage: new LibSQLStore({
      id: 'agent-memory',
      url: 'file:../mastra.db',
    }),
  }),
});
