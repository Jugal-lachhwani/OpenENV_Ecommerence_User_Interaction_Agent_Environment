"""Compatibility module for legacy imports.

The production environment implementation now lives in server/environment.py.
"""

from .environment import EcommerceCustomerInteractionEnvironment

MyEnvironment = EcommerceCustomerInteractionEnvironment
