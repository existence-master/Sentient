{#- towncrier template: one "### Type" list per kind of change, in Keep a Changelog style.
    The issue number goes before the closing full stop, like the hand-written entries:
    "- **Pausing a task stops it now.** Before, ... (#405)." -#}
{% for section, _ in sections.items() %}
{% for category, val in definitions.items() if category in sections[section] %}

### {{ definitions[category]['name'] }}
{% for text, values in sections[section][category].items() %}
{% if values and text[-1:] in ['.', '!', '?'] %}
- {{ text[:-1] }} ({{ values | join(', ') }}){{ text[-1] }}
{% elif values %}
- {{ text }} ({{ values | join(', ') }})
{% else %}
- {{ text }}
{% endif %}
{% endfor %}
{% endfor %}
{% endfor %}
