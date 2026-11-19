"""Provider API keys for the live round trips that triage and review sessions write and run.

Each key comes from a repo secret that the workflow passes only to the step starting the
session. write_provider_keys writes it to the factory-providers vault as an environment_variable
credential: the sandbox sees a placeholder, and the real key is injected only into request
headers sent to that provider's API host. An unset secret deletes its credential, so removing the
repo secret turns the key off for every later session.
"""

import os
from typing import Any, NamedTuple

import anthropic

PROVIDERS_VAULT_LOCK_KEY = "./vaults/providers.yaml"
PROVIDERS_VAULT_NAME = "factory-providers"


class ProviderKey(NamedTuple):
    secret_name: str  # the variable the sandbox sees
    source: str  # the runner variable that holds the repo secret
    host: str  # the only host the real key is sent to


PROVIDER_KEYS = (
    ProviderKey("ANTHROPIC_API_KEY", "FACTORY_SANDBOX_ANTHROPIC_API_KEY", "api.anthropic.com"),
)


def write_provider_keys(*, client: anthropic.Anthropic, lock: dict[str, Any]) -> list[str]:
    """Sync the repo's provider keys into the providers vault; return its ID if it holds any."""
    values = {key: os.environ.get(key.source, "") for key in PROVIDER_KEYS}
    resource = lock.get("resources", {}).get(PROVIDERS_VAULT_LOCK_KEY)
    if resource is None:
        if any(values.values()):
            print(
                f"::warning::{PROVIDERS_VAULT_LOCK_KEY} is not in claude-lock.json yet; run Factory Sync."
            )
        return []
    vault_id = resource["id"]
    # claude-lock.json lives on a branch a session's token can push, and triage sessions get this
    # vault, so an ID naming any other vault, such as the one holding write tokens, stops the run.
    vault_name = client.beta.vaults.retrieve(vault_id).display_name
    if vault_name != PROVIDERS_VAULT_NAME:
        raise SystemExit(
            f"::error::claude-lock.json maps {PROVIDERS_VAULT_LOCK_KEY} to {vault_name!r}, not "
            f"{PROVIDERS_VAULT_NAME!r}. Check the factory-state branch's history, and run "
            "Factory Sync to rewrite it."
        )

    existing: dict[str, list[str]] = {}
    for credential in client.beta.vaults.credentials.list(vault_id):
        if credential.auth.type == "environment_variable":
            existing.setdefault(credential.auth.secret_name, []).append(credential.id)

    written = False
    for key, value in values.items():
        ids = existing.get(key.secret_name, [])
        if value and key.host == "api.anthropic.com":
            check_separate_workspace(key=key, value=value, vault_id=vault_id)
        if value and ids:
            client.beta.vaults.credentials.update(
                ids[0],
                vault_id=vault_id,
                auth={
                    "type": "environment_variable",
                    "secret_value": value,
                    "networking": {"type": "limited", "allowed_hosts": [key.host]},
                    "injection_location": {"header": True},
                },
            )
        elif value:
            client.beta.vaults.credentials.create(
                vault_id,
                display_name=f"{key.secret_name} for {key.host}",
                auth={
                    "type": "environment_variable",
                    "secret_name": key.secret_name,
                    "secret_value": value,
                    "networking": {"type": "limited", "allowed_hosts": [key.host]},
                    "injection_location": {"header": True},
                },
            )
        # Two runs racing to create the credential leave a duplicate; keep one.
        for credential_id in ids[1:] if value else ids:
            client.beta.vaults.credentials.delete(credential_id, vault_id=vault_id)
        written = written or bool(value)
    return [vault_id] if written else []


def check_separate_workspace(*, key: ProviderKey, value: str, vault_id: str) -> None:
    """Stop the run if an Anthropic key for the sandbox can see this workspace's vaults.

    The sandbox sends the key to api.anthropic.com, which also serves the Managed Agents API, so a
    key from the factory's workspace would let a session list its vaults and start a session with
    the write-token one. A key from another workspace gets a 404 for this vault.
    """
    try:
        anthropic.Anthropic(api_key=value).beta.vaults.retrieve(vault_id)
    except (anthropic.NotFoundError, anthropic.PermissionDeniedError):
        return
    except anthropic.AuthenticationError:
        raise SystemExit(
            f"::error::Anthropic rejected {key.source}; replace the repo secret."
        ) from None
    raise SystemExit(
        f"::error::{key.source} belongs to the factory's own workspace. Use a key from a separate, "
        "spend-limited workspace; see Provider keys in factory/README.md."
    )
