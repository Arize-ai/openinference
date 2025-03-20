import "./instrumentation";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

async function main() {
  const tool = new OpenMeteoTool();
  const result = await tool.run({
    location: { name: "New York" },
    start_date: "2024-10-10",
    end_date: "2024-10-10",
  });
  // eslint-disable-next-line no-console
  console.log(result.getTextContent());
}

// eslint-disable-next-line no-console
main().catch(console.error);
