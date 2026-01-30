# src/structure/ast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Optional

# ---- AST node types ----

@dataclass(frozen=True)
class Node:
    pass

@dataclass(frozen=True)
class Row(Node):
    items: List[Node]

@dataclass(frozen=True)
class Mi(Node):
    text: str  # identifiers (a, x, z, \alpha)

@dataclass(frozen=True)
class Mn(Node):
    text: str  # numbers

@dataclass(frozen=True)
class Mo(Node):
    text: str  # operators

@dataclass(frozen=True)
class MText(Node):
    text: str  # fallback / text nodes

@dataclass(frozen=True)
class Frac(Node):
    num: Node
    den: Node

@dataclass(frozen=True)
class Sqrt(Node):
    body: Node

@dataclass(frozen=True)
class Sup(Node):
    base: Node
    exp: Node

@dataclass(frozen=True)
class Sub(Node):
    base: Node
    sub: Node

@dataclass(frozen=True)
class SubSup(Node):
    base: Node
    sub: Node
    sup: Node

@dataclass(frozen=True)
class Fenced(Node):
    open: str
    body: Node
    close: str


AST = Union[
    Row, Mi, Mn, Mo, MText, Frac, Sqrt, Sup, Sub, SubSup, Fenced
]