from pathlib import Path

import yaml

BASE_CONFIGS = [
    {"package-ecosystem": "gradle", "directory": "/java"},
    {"package-ecosystem": "npm", "directory": "/js"},
    {"package-ecosystem": "pip", "directory": "/python"},
]

IGNORE_UPDATE_TYPES = [
    "version-update:semver-major",
    "version-update:semver-minor",
    "version-update:semver-patch",
]


def _find_package_directories(root: Path, prefix: str, manifest: str) -> list[str]:
    return sorted(
        str(path)
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and (path / manifest).exists()
    )


def _build_update_config(package_ecosystem: str, directory: str) -> dict[str, object]:
    return {
        "package-ecosystem": package_ecosystem,
        "directory": directory,
        "schedule": {"interval": "weekly"},
        "ignore": [
            {
                "dependency-name": "*",
                "update-types": list(IGNORE_UPDATE_TYPES),
            }
        ],
    }


def generate_dependabot_config() -> str:
    updates = [
        _build_update_config(cfg["package-ecosystem"], cfg["directory"])
        for cfg in BASE_CONFIGS
    ]

    updates.extend(
        _build_update_config("pip", directory)
        for directory in _find_package_directories(
            Path("python/instrumentation"),
            "openinference-instrumentation-",
            "pyproject.toml",
        )
    )
    updates.extend(
        _build_update_config("npm", directory)
        for directory in _find_package_directories(
            Path("js/packages"),
            "openinference-",
            "package.json",
        )
    )

    return yaml.safe_dump({"version": 2, "updates": updates}, default_flow_style=False)


if __name__ == "__main__":
    print(generate_dependabot_config(), end="")
