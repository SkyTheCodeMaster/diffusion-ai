# Handle authentication and request verification
from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import TYPE_CHECKING

from aiohttp.web import Response

if TYPE_CHECKING:
  from aiohttp import ClientSession

  from .extra_request import Application, Request


class Approval(Enum):
  DEFAULT = 0
  APPROVED = 1
  PENDING = 2
  DENIED = 3


class User:
  username: str
  super_admin: bool
  email: str
  token: str

  def __init__(
    self, *, username: str, super_admin: bool, email: str, token: str
  ) -> None:
    self.username = username
    self.super_admin = super_admin
    self.email = email
    self.token = token


class Key:
  name: str
  id: str
  data: str
  user: User
  project: Project

  def __init__(
    self,
    *,
    name: str = None,
    id: str = None,
    data: str = None,
    user: User = None,
    project: Project = None,
  ) -> None:
    self.name = (name,)
    self.id = id
    self.data = data
    self.user = user
    self.project = project


class Project:
  id: int
  name: str
  public: bool
  open: bool
  url: str

  def __init__(
    self,
    *,
    id: int = None,
    name: str = None,
    public: bool = None,
    open: bool = None,
    url: str = None,
    description: str = None,
  ) -> None:
    self.id = id
    self.name = name
    self.public = public
    self.open = open
    self.url = url
    self.description = description


# Hashed auth token -> user, expiry
auth_cache: dict[str, tuple[User, int]] = {}


# All this does is authenticate a user existing.
async def authenticate(
  request: Request, *, cs: ClientSession = None, use_cache=True
) -> User | Response | Key:
  app: Application = request.app
  if cs is None:
    cs = app.cs

  # We prioritize header authentication over cookie authentication.
  auth_token = request.cookies.get("Authorization", None)
  auth_token = request.headers.get("Authorization", auth_token)

  if auth_token is None:
    return Response(
      status=401, body="pass Authorization header or Authorization cookie."
    )

  auth_token = auth_token.removeprefix("Bearer ")

  # Get the hash of it for cache checking
  if use_cache:
    token_hash = hashlib.sha512(auth_token.encode()).hexdigest()
    if token_hash in auth_cache:
      details = auth_cache[token_hash]
      current_time = time.time()
      if current_time < details[1]:
        return details[0]

  # Now we want to send the request
  # Removing and adding Bearer is mildly redundant
  headers = {"Authorization": f"Bearer {auth_token}"}

  async with cs.get(
    "https://auth.skystuff.cc/api/user/get/", headers=headers
  ) as resp:
    if resp.status == 200:
      data = json.loads(await resp.text())
      u = User(
        username=data["name"],
        super_admin=data["super_admin"],
        email=data["email"],
        token=data["token"],
      )

      token_hash = hashlib.sha512(auth_token.encode()).hexdigest()
      current_time = time.time()
      auth_cache[token_hash] = (u, current_time + 600)

      return u
    elif resp.status == 400:
      text = await resp.text()
      if not text.startswith("please use /key/"):
        return Response(status=401, body="invalid token")
      async with cs.get(
        "https://auth.skystuff.cc/api/key/" + auth_token
      ) as kresp:
        if kresp.status != 200:
          return Response(status=401, body="invalid token")
        data = json.loads(await kresp.text())
        project = Project(**data["project"])
        user = User(**data["user"])
        key = Key(
          name=data["name"],
          id=data["id"],
          data=data["data"],
          user=user,
          project=project,
        )
        return key

    else:
      return Response(status=401, body="invalid token")


async def get_project_status(
  user: User, project_name: str, *, cs: ClientSession
) -> Approval | False:
  headers = {"Authorization": f"Bearer {user.token}"}
  async with cs.get(
    f"https://auth.skystuff.cc/api/project/status/{project_name}",
    headers=headers,
  ) as resp:
    if resp.status != 200:
      return False
    data = await resp.json()
    approval = Approval[data["approval"].upper()]
    return approval
