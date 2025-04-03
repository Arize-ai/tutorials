import readline from "node:readline/promises";
import "./instrumentation";
import { stdin, stdout } from "node:process";
import picocolors from "picocolors";
import * as R from "remeda";
import stripAnsi from "strip-ansi";
import type { Abortable } from "node:events";

interface ReadFromConsoleInput {
  fallback?: string;
  input?: string;
  allowEmpty?: boolean;
}

export function createConsoleReader({
  fallback,
  input = "User 👤 : ",
  allowEmpty = false,
}: ReadFromConsoleInput = {}) {
  const rl = readline.createInterface({ input: stdin, output: stdout, terminal: true, prompt: "" });
  let isActive = true;

  return {
    write(role: string, data: string) {
      rl.write(
        [role && R.piped(picocolors.red, picocolors.bold)(role), stripAnsi(data ?? "")]
          .filter(Boolean)
          .join(" ")
          .concat("\n"),
      );
    },

    async prompt(): Promise<string> {
      for await (const { prompt } of this) {
        return prompt;
      }
      process.exit(0);
    },

    async askSingleQuestion(queryMessage: string, options?: Abortable): Promise<string> {
      const answer = await rl.question(
        R.piped(picocolors.cyan, picocolors.bold)(queryMessage),
        options ?? { signal: undefined },
      );
      return stripAnsi(answer.trim());
    },

    close() {
      stdin.pause();
      rl.close();
    },

    async *[Symbol.asyncIterator]() {
      if (!isActive) {
        return;
      }

      try {
        rl.write(
          `${picocolors.dim(`Interactive session has started. To escape, input 'q' and submit.\n`)}`,
        );

        for (let iteration = 1, prompt = ""; isActive; iteration++) {
          prompt = await rl.question(R.piped(picocolors.cyan, picocolors.bold)(input));
          prompt = stripAnsi(prompt);

          if (prompt === "q") {
            break;
          }
          if (!prompt.trim() || prompt === "\n") {
            prompt = fallback ?? "";
          }
          if (allowEmpty !== false && !prompt.trim()) {
            rl.write("Error: Empty prompt is not allowed. Please try again.\n");
            iteration -= 1;
            continue;
          }
          yield { prompt, iteration };
        }
      } catch (e) {
        if (e.code === "ERR_USE_AFTER_CLOSE") {
          return;
        }
        throw e;
      } finally {
        isActive = false;
        rl.close();
      }
    },
  };
}



import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow } from "beeai-framework/workflows/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const workflow = new AgentWorkflow("Smart assistant");
const llm = new OllamaChatModel("llama3.2");

workflow.addAgent({
  name: "Researcher",
  role: "A diligent researcher",
  instructions: "You look up and provide information about a specific topic.",
  tools: [new WikipediaTool()],
  llm,
});
workflow.addAgent({
  name: "WeatherForecaster",
  role: "A weather reporter",
  instructions: "You provide detailed weather reports.",
  tools: [new OpenMeteoTool()],
  llm,
});
workflow.addAgent({
  name: "DataSynthesizer",
  role: "A meticulous and creative data synthesizer",
  instructions: "You can combine disparate information into a final coherent summary.",
  llm,
});

const reader = createConsoleReader();
reader.write("Assistant 🤖 : ", "What location do you want to learn about?");
for await (const { prompt } of reader) {
  const { result } = await workflow
    .run([
      { prompt: "Provide a short history of the location.", context: prompt },
      {
        prompt: "Provide a comprehensive weather summary for the location today.",
        expectedOutput:
          "Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
      },
      {
        prompt: "Summarize the historical and weather data for the location.",
        expectedOutput:
          "A paragraph that describes the history of the location, followed by the current weather conditions.",
      },
    ])
    .observe((emitter) => {
      emitter.on("success", (data) => {
        reader.write(
          `Step '${data.step}' has been completed with the following outcome:\n`,
          data.state?.finalAnswer ?? "-",
        );
      });
    });

  reader.write(`Assistant 🤖`, result.finalAnswer);
  reader.write("Assistant 🤖 : ", "What location do you want to learn about?");
}


