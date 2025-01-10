from io import BytesIO

import requests
from PIL import Image
from smolagents import CodeAgent, GradioUI, HfApiModel, Tool
from smolagents.default_tools import VisitWebpageTool


class GetCatImageTool(Tool):
    name = "get_cat_image"
    description = "Get a cat image"
    inputs = {}
    output_type = "image"

    def __init__(self):
        super().__init__()
        self.url = "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"

    def forward(self):
        response = requests.get(self.url)

        return Image.open(BytesIO(response.content))


get_cat_image = GetCatImageTool()

agent = CodeAgent(
    tools=[get_cat_image, VisitWebpageTool()],
    model=HfApiModel(),
    additional_authorized_imports=["Pillow", "requests", "markdownify"],  # "duckduckgo-search",
    use_e2b_executor=True,
)

agent.run(
    "Return me an image of a cat. Directly use the image provided in your state.",
    additional_args={"cat_image": get_cat_image()},
)

GradioUI(agent).launch()
