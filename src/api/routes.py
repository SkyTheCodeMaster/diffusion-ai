from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.web import Response
from utils.diffusion import generate
from utils.cors import add_cors_routes
from utils.limiter import Limiter

if TYPE_CHECKING:
  from utils.extra_request import Request

with open("config.toml") as f:
  config = tomllib.loads(f.read())
  frontend_version = config["pages"]["frontend_version"]
  exempt_ips = config["srv"]["ratelimit_exempt"]
  api_version = config["srv"]["api_version"]

limiter = Limiter(exempt_ips=exempt_ips)
routes = web.RouteTableDef()

@routes.get("/srv/get/")
@limiter.limit("60/m")
async def get_lp_get(request: Request) -> Response:
  packet = {
    "frontend_version": frontend_version,
    "api_version": api_version,
  }

  if request.app.POSTGRES_ENABLED:
    database_size_record = await request.conn.fetchrow("SELECT pg_size_pretty ( pg_database_size ( current_database() ) );")
    packet["db_size"] = database_size_record.get("pg_size_pretty","-1 kB")

  return web.json_response(packet)

@routes.post("/diffusion/")
@limiter.limit("10/m")
async def post_diffusion(request: Request) -> Response:
  body = await request.json()

  if "prompt" not in body:
    return Response(status=400,text="must include prompt in body")

  prompt = body.get("prompt")
  negative_prompt = body.get("negative_prompt", None)
  width = body.get("width", None)
  height = body.get("height", None)

  try:
    result = await generate(prompt, negative_prompt=negative_prompt, width=width, height=height)
  except Exception as e:
    return Response(status=500, text=str(e))
  
  return web.json_response(result)


async def setup(app: web.Application) -> None:
  for route in routes:
    app.LOG.info(f"  ↳ {route}")
  app.add_routes(routes)
  add_cors_routes(routes, app)