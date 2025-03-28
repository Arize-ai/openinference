import json
import re
import sys


def main() -> None:
    assert (
        num_arguments := len(sys.argv)
    ) == 3, f"Script requires two arguments, but received {num_arguments}"
    diff_files = json.loads(sys.argv[1])
    tox_environments = json.loads(sys.argv[2])

    selected_environments = select_tox_environments(diff_files, tox_environments)
    print(json.dumps(selected_environments))


def select_tox_environments(
    diff_files: list[str],
    tox_environments: list[str],
) -> list[str]:
    semconv_has_diff, instrumentation_has_diff, instrumentors_with_diff = find_diffs(diff_files)
    semconv_environments, instrumentation_environments, instrumentor_environments = (
        split_tox_environments(tox_environments)
    )

    if semconv_has_diff:
        return semconv_environments + instrumentation_environments + instrumentor_environments
    if instrumentation_has_diff:
        return instrumentation_environments + instrumentor_environments

    selected_environments = []
    for instrumentor in instrumentors_with_diff:
        selected_environments.extend(
            [env for env in instrumentor_environments if instrumentor in env]
        )
    return selected_environments


def find_diffs(diff_files: list[str]) -> tuple[bool, bool, set[str]]:
    semconv_has_diff = any(
        file.startswith("python/openinference-semantic-conventions/") for file in diff_files
    )
    instrumentation_has_diff = any(
        file.startswith("python/openinference-instrumentation/") for file in diff_files
    )
    instrumentors_with_diff = set()
    for diff_file in diff_files:
        if match := re.match(
            "^python/instrumentation/openinference-instrumentation-([^/]+)/.+$", diff_file
        ):
            instrumentor_name = match.group(1).replace("-", "_")
            instrumentors_with_diff.add(instrumentor_name)
    return (
        semconv_has_diff,
        instrumentation_has_diff,
        instrumentors_with_diff,
    )


def split_tox_environments(
    tox_environments: list[str],
) -> tuple[list[str], list[str], list[str]]:
    semconv_environments = []
    instrumentation_environments = []
    other_environments = []
    for environment in tox_environments:
        if "semconv" in environment:
            semconv_environments.append(environment)
        elif "instrumentation" in environment:
            instrumentation_environments.append(environment)
        else:
            other_environments.append(environment)
    return semconv_environments, instrumentation_environments, other_environments


if __name__ == "__main__":
    main()
