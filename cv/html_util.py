"""HTML escaping for unsafe_allow_html blocks."""

import html


def escape_html(text: object) -> str:
    return html.escape(str(text), quote=True)
