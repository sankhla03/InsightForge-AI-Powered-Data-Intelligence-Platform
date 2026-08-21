from django import template

register = template.Library()


@register.filter
def dict_get(d, key):
    """Access dictionary key in Django template"""
    return d.get(key, '')

