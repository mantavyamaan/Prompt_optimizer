from __future__ import annotations

import hashlib
import subprocess

from jinja2 import Environment, StrictUndefined

from .models import PromptModules

_ENV = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
_TEMPLATE = _ENV.from_string("""{{ role }}

{{ context_rules }}

{{ format_instructions }}
{% if few_shot_examples %}

Examples:
{{ few_shot_examples }}
{% endif %}

{{ constraints }}

Input:
{{ user_input }}
""")


def compile_prompt(modules: PromptModules, variables: dict[str, object]) -> str:
    """Render with StrictUndefined so missing inputs cannot silently degrade a prompt."""
    return _TEMPLATE.render(**modules.model_dump(), **variables).strip()


def compiled_hash(modules: PromptModules) -> str:
    canonical = "\x1f".join(getattr(modules, name) for name in modules.model_fields)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def code_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"
