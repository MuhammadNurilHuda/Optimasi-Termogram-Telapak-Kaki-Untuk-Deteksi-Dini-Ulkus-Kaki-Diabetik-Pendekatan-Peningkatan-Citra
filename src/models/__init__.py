# src/models/__init__.py

from .model1 import create_model as create_model1
from .model2 import create_model as create_model2
from .model3 import create_model as create_model3
from .model4 import create_model as create_model4

__all__ = ['create_model1', 'create_model2', 'create_model3', 'create_model4']