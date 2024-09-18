import json
import re
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import Iterator

from langsmith.schemas import LangSmithInfo
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from uvicorn import Config, Server

OUTPUT_FILE_NAME = "langsmith_data_capture"


class _Receiver(Server):
    def _new_file(self) -> Path:
        filename = f"{OUTPUT_FILE_NAME}"
        filename_pattern = re.escape(OUTPUT_FILE_NAME) + r"(\s+\((?P<file_no>\d+)\))?\.txt$"
        output_directory = Path().home() / "langsmith_data"
        output_directory.mkdir(exist_ok=True)
        max_file_no = max(
            (
                int(m.groupdict(default="0")["file_no"])
                for f in output_directory.iterdir()
                if (m := re.search(filename_pattern, str(f)))
            ),
            default=-1,
        )
        if max_file_no > -1:
            filename += f" ({max_file_no + 1})"
        return output_directory / f"{filename}.txt"

    async def _capture(self, request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse(LangSmithInfo().dict())
        timestamp = datetime.now().isoformat()
        body = await request.json()
        self._buf.write(f"{timestamp}\n\n")
        self._buf.write(f"{request.method} {request.scope.get('path', '')}\n\n")
        self._buf.write(f"{json.dumps(body, indent=4, sort_keys=True)}\n\n")
        self._buf.write(f"{'🤖' + '=' * 50}\n\n")
        return Response()

    async def _shutdown(self) -> None:
        with open(self._new_file(), "w") as f:
            f.write(self._buf.getvalue())
        self._buf.close()

    def __init__(self, port: int) -> None:
        self._buf = StringIO()
        route = Route("/{path:path}", self._capture, methods=["POST", "PATCH", "GET"])
        app = Starlette(routes=[route], on_shutdown=[self._shutdown])
        config = Config(app=app, port=port)
        super().__init__(config=config)

    def install_signal_handlers(self) -> None:
        pass

    @contextmanager
    def run_in_thread(self) -> Iterator[Thread]:
        """A coroutine to keep the server running in a thread."""
        thread = Thread(target=self.run)
        thread.start()
        time_limit = time() + 5  # 5 seconds
        try:
            while not self.started and thread.is_alive() and time() < time_limit:
                sleep(1e-3)
            if time() > time_limit:
                raise RuntimeError("server took too long to start")
            yield thread
        finally:
            self.should_exit = True
            thread.join(timeout=5)
