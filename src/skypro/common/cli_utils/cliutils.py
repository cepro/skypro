from typing import Dict


def substitute_vars(string: str, variables: Dict[str, str]) -> str:
    """
    Replaces the $variables that are present in `string` with their associated value in the `variables` dictionary.
    For example, "$DIR/one/two" might have $DIR substituted with a path.
    """
    for key, val in variables.items():
        string = string.replace(f"${key}", val)

    return string

