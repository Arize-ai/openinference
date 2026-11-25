"""Short-lived Anthropic access tokens for the live round trips that triage and review run.

No provider key is stored anywhere. The workflow job proves who it is with its GitHub OIDC token
and exchanges it through Workload Identity Federation for an Anthropic access token that lives
about ten minutes. SandboxAnthropicToken writes that token to the factory-providers vault as
ANTHROPIC_AUTH_TOKEN, injected only into headers sent to api.anthropic.com, so the sandbox sees a
placeholder and its SDK sends it as a Bearer token. The watch step mints a new one before the old
one expires, so a session's token dies minutes after its job ends. The federation rule's
`workspace:inference` scope limits the token to the Messages API; see Provider keys in
factory/README.md.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import anthropic

PROVIDERS_VAULT_LOCK_KEY = "./vaults/providers.yaml"
PROVIDERS_VAULT_NAME = "factory-providers"
SECRET_NAME = "ANTHROPIC_AUTH_TOKEN"
API_HOST = "api.anthropic.com"
TOKEN_URL = f"https://{API_HOST}/v1/oauth/token"
AUDIENCE = f"https://{API_HOST}"
# Repo variables naming the federation rule; none of them is a secret.
ORGANIZATION_VARIABLE = "FACTORY_SANDBOX_ANTHROPIC_ORGANIZATION_ID"
RULE_VARIABLE = "FACTORY_SANDBOX_ANTHROPIC_FEDERATION_RULE_ID"
SERVICE_ACCOUNT_VARIABLE = "FACTORY_SANDBOX_ANTHROPIC_SERVICE_ACCOUNT_ID"
WORKSPACE_VARIABLE = "FACTORY_SANDBOX_ANTHROPIC_WORKSPACE_ID"
# Tokens live about ten minutes and the watch loop polls every minute, so re-minting with five
# minutes left keeps the vault's token valid through several missed polls.
REFRESH_MARGIN_SECONDS = 300
MAX_ATTEMPTS = 3


def read_providers_vault_id(*, client: anthropic.Anthropic, lock: dict[str, Any]) -> str | None:
    """Return the providers vault's ID once Factory Sync has created it.

    claude-lock.json lives on a branch a session's token can push, and triage sessions get this
    vault, so an ID naming any other vault, such as the one holding write tokens, stops the run.
    """
    resource = lock.get("resources", {}).get(PROVIDERS_VAULT_LOCK_KEY)
    if resource is None:
        return None
    vault_name = client.beta.vaults.retrieve(resource["id"]).display_name
    if vault_name != PROVIDERS_VAULT_NAME:
        raise SystemExit(
            f"::error::claude-lock.json maps {PROVIDERS_VAULT_LOCK_KEY} to {vault_name!r}, not "
            f"{PROVIDERS_VAULT_NAME!r}. Check the factory-state branch's history, and run "
            "Factory Sync to rewrite it."
        )
    return str(resource["id"])


def post_json(*, url: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_github_oidc_token() -> str:
    """A fresh GitHub OIDC token for this job; each exchange needs one, since its jti is single-use."""
    url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    request = urllib.request.Request(
        f"{url}&audience={urllib.parse.quote(AUDIENCE, safe='')}",
        headers={"Authorization": f"bearer {os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return str(json.load(response)["value"])


class SandboxAnthropicToken:
    def __init__(self, *, client: anthropic.Anthropic, lock: dict[str, Any]) -> None:
        self.client = client
        self.vault_id = read_providers_vault_id(client=client, lock=lock)
        self.exchange = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "organization_id": os.environ.get(ORGANIZATION_VARIABLE, ""),
            "federation_rule_id": os.environ.get(RULE_VARIABLE, ""),
            **{
                field: os.environ[variable]
                for field, variable in [
                    ("service_account_id", SERVICE_ACCOUNT_VARIABLE),
                    ("workspace_id", WORKSPACE_VARIABLE),
                ]
                if os.environ.get(variable)
            },
        }
        self.configured = bool(
            self.exchange["organization_id"] and self.exchange["federation_rule_id"]
        )
        self.expires_at = 0.0
        self.scope_checked = False

    def vault_ids(self) -> list[str]:
        """Write the session's first token and return the vault to attach, or [] without one."""
        if self.vault_id is None:
            if self.configured:
                print(
                    f"::warning::{PROVIDERS_VAULT_LOCK_KEY} is not in claude-lock.json yet; run Factory Sync."
                )
            return []
        if not self.configured:
            # Unsetting the repo variables turns the round trip off: drop any token left behind.
            for credential_id in self.list_credential_ids():
                self.client.beta.vaults.credentials.delete(credential_id, vault_id=self.vault_id)
            return []
        if "ACTIONS_ID_TOKEN_REQUEST_URL" not in os.environ:
            raise SystemExit(
                "::error::The job needs `permissions: id-token: write` to mint the sandbox's Anthropic token."
            )
        self.refresh()
        return [self.vault_id]

    def refresh(self) -> None:
        """Mint and write a new token when the current one has less than the margin left."""
        if self.vault_id is None or not self.configured:
            return
        if time.time() < self.expires_at - REFRESH_MARGIN_SECONDS:
            return
        token, expires_in = self.mint()
        if not self.scope_checked:
            self.check_messages_only(token=token)
            self.scope_checked = True
        self.write(token=token)
        self.expires_at = time.time() + expires_in

    def refresh_quietly(self) -> None:
        """refresh() for the watch loop: a failed mint costs the session its round trip, not its report."""
        try:
            self.refresh()
        except Exception as error:  # noqa: BLE001 - any failure here must not end the watch
            print(f"::warning::Could not refresh the sandbox's Anthropic token: {error}")

    def mint(self) -> tuple[str, float]:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = post_json(
                    url=TOKEN_URL, body={**self.exchange, "assertion": fetch_github_oidc_token()}
                )
                return str(response["access_token"]), float(response["expires_in"])
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == MAX_ATTEMPTS - 1:
                    # Anthropic answers a denied exchange with an opaque 401; the reason, such as
                    # a rule condition that didn't match, is in the Console's authentication history.
                    detail = error.read().decode(errors="replace")
                    raise RuntimeError(f"Token exchange returned {error.code}: {detail}") from error
            except urllib.error.URLError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
            time.sleep(5 * 2**attempt)
        raise AssertionError("unreachable")

    def check_messages_only(self, *, token: str) -> None:
        """Stop the run unless the token is refused the Managed Agents API.

        api.anthropic.com serves Managed Agents as well as Messages. Only the `workspace:inference`
        scope is refused vaults, so a 403 here proves the sandbox can't reach the factory's
        vaults or start agents of its own. An explicit token keeps the SDK from also sending the
        job's ANTHROPIC_API_KEY.
        """
        try:
            anthropic.Anthropic(auth_token=token).beta.vaults.list(limit=1)
        except anthropic.PermissionDeniedError:
            return
        raise SystemExit(
            "::error::The sandbox's Anthropic token can call the Managed Agents API. Set the "
            "federation rule's scope to workspace:inference; see Provider keys in "
            "factory/README.md."
        )

    def list_credential_ids(self) -> list[str]:
        assert self.vault_id is not None
        return [
            credential.id
            for credential in self.client.beta.vaults.credentials.list(self.vault_id)
            if credential.auth.type == "environment_variable"
            and credential.auth.secret_name == SECRET_NAME
        ]

    def write(self, *, token: str) -> None:
        """Update the vault's token credential, or create it. Concurrent jobs share it, and each
        writes a fresh token, so the latest write is always one that hasn't expired."""
        assert self.vault_id is not None
        ids = self.list_credential_ids()
        if ids:
            self.client.beta.vaults.credentials.update(
                ids[0],
                vault_id=self.vault_id,
                auth={
                    "type": "environment_variable",
                    "secret_value": token,
                    "networking": {"type": "limited", "allowed_hosts": [API_HOST]},
                    "injection_location": {"header": True},
                },
            )
        else:
            self.client.beta.vaults.credentials.create(
                self.vault_id,
                display_name=f"{SECRET_NAME} for {API_HOST}",
                auth={
                    "type": "environment_variable",
                    "secret_name": SECRET_NAME,
                    "secret_value": token,
                    "networking": {"type": "limited", "allowed_hosts": [API_HOST]},
                    "injection_location": {"header": True},
                },
            )
        # Two jobs racing to create the credential leave a duplicate; keep one.
        for credential_id in ids[1:]:
            self.client.beta.vaults.credentials.delete(credential_id, vault_id=self.vault_id)
