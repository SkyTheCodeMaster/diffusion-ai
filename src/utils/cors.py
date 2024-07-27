from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp.web import Response, middleware, options

if TYPE_CHECKING:
  from aiohttp.web import Application, Request, Response, RouteTableDef


def add_cors(request: Request, response: Response) -> None:
  if response.headers.get("Access-Control-Allow-Origin", None) is None:
    response.headers.add("Access-Control-Allow-Origin", request.headers["Origin"])
  if response.headers.get("Access-Control-Allow-Credentials", None) is None:
    response.headers.add("Access-Control-Allow-Credentials", "true")
  if response.headers.get("Access-Control-Allow-Methods", None) is None:
    response.headers.add("Access-Control-Allow-Methods", "GET")
  if response.headers.get("Access-Control-Allow-Headers", None) is None:
    response.headers.add("Access-Control-Allow-Headers", "Authorization")

@middleware
async def cors_middleware(request: Request, handler):
  resp = await handler(request)
  try:
    add_cors(request, resp)
  except Exception:
    pass
  return resp

def add_cors_routes(routes: RouteTableDef, app: Application) -> None:
  to_add = []
  for route in routes:
    to_add.append(options(route.path, handle_options))
  app.add_routes(to_add)


async def handle_options(request: Request) -> Response:
  resp = Response()
  try:
    add_cors(request, resp)
  except Exception:
    pass

  return resp
