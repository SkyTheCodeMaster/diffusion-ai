from __future__ import annotations

import os
import pathlib
import tomllib
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
  pass

with open("config.toml") as f:
  config = tomllib.loads(f.read())

routes = web.RouteTableDef()

# Load all of the templates in the static folder.
templates: dict[str,str] = {}
sup_templates: dict[str,str] = {}

def join(a: str, b: str) -> str:
  "Join 2 filepaths"
  return str(pathlib.Path(a).joinpath(pathlib.Path(b)))

for root, dirs, files in os.walk("frontend/templates"):
  for file in files:
    filepath = join(root,file)
    with open(filepath,"r") as f:
      templates[filepath.removeprefix("frontend/templates").removeprefix("/")] = f.read()

for root, dirs, files in os.walk("frontend/supporting"):
  for file in files:
    filepath = join(root,file)
    with open(filepath,"r") as f:
      sup_templates[filepath.removeprefix("frontend/supporting").removeprefix("/")] = f.read()

for name, contents in templates.items():
  async def serve(request: web.Request, name=name, contents=contents) -> web.Response:
    return web.Response(text=contents, content_type="text/html")

  clean_file_name = name.replace("/","_").removesuffix(".html")
  serve.__name__ = f"get_{clean_file_name}"
  serve.__qualname__ = serve.__name__

  serve_name = name.removesuffix(".html")

  routes._items.append(web.RouteDef("GET",f"/{serve_name}", serve, {}))

for name, contents in sup_templates.items():
  async def serve(request: web.Request, name=name, contents=contents) -> web.Response:
    return web.Response(text=contents, content_type="text/html")

  clean_file_name = name.replace("/","_").removesuffix(".html")
  serve.__name__ = f"get_{clean_file_name}"
  serve.__qualname__ = serve.__name__

  serve_name = name.removesuffix(".html")

  routes._items.append(web.RouteDef("GET",f"/sup/{serve_name}", serve, {}))

@routes.get("/")
async def get_index(request: web.Request) -> web.Response:
  # Check devmode

  rendered = templates["index.html"]
  return web.Response(body=rendered,content_type="text/html")

routes._items.append(web.static("/","frontend/static"))

async def setup(app: web.Application) -> None:
  for route in routes:
    app.LOG.info(f"  ↳ {route}")
  app.add_routes(routes)