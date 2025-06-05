import os
from typing import Annotated, Dict

from marshmallow import fields
from skypro.common.cli_utils.cli_utils import substitute_vars


class PathField(fields.Field):
    """
    This Marshmallow field type is used to deserialize file paths. It expands the local user tilde symbol and also
    substitutes variables (in the $VAR_NAME format).
    The variables must be first set on the `vars_for_substitution` class variable before deserializing.
    """

    vars_for_substitution = {}  # class variable defines any variables for substitution into the paths

    def _serialize(self, value, attr, obj, **kwargs):
        raise NotImplementedError("Serialization not yet defined")

    def _deserialize(self, value, attr, data, **kwargs):
        # Expand any `~/` syntax and $ENV_VARS that are used
        return resolve_file_path(value, PathField.vars_for_substitution)


# The marshmallow_dataclass library doesn't use the PathField directly, but instead needs a Type defining:
PathType = Annotated[str, PathField]


def resolve_file_path(file: str, env_vars: Dict) -> str:
    """
    Function to substitutes env vars and resolve `~` in file paths.
    This is used by the above PathField class when a path is specified in the Marshmallow configuration. But it is also
    needed by the rates parsing code in the simt-common library which doesn't currently use Marshmallow.
    """
    return os.path.expanduser(substitute_vars(file, env_vars))
