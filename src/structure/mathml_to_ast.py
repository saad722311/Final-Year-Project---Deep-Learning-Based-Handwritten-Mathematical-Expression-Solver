# src/structure/mathml_to_ast.py
from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import List, Optional

from src.structure.ast import (
    Node, Row, Mi, Mn, Mo, MText,
    Frac, Sqrt, Sup, Sub, SubSup, Fenced
)

def _local(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def _text(el: ET.Element) -> str:
    # element.text sometimes None
    return (el.text or "").strip()

def _children(el: ET.Element) -> List[ET.Element]:
    return list(el)

def mathml_to_ast(math_el: ET.Element) -> Node:
    """
    Accepts <math> root element, returns AST Node.
    """
    # Typically <math><mrow>...</mrow></math>, but not always.
    kids = _children(math_el)
    if len(kids) == 0:
        return MText(_text(math_el))
    if len(kids) == 1:
        return _convert(kids[0])
    return Row([_convert(k) for k in kids])

def _convert(el: ET.Element) -> Node:
    tag = _local(el.tag)

    if tag in ("mrow", "math"):
        items = [_convert(c) for c in _children(el)]
        # if empty, fallback to text
        if not items:
            t = _text(el)
            return MText(t) if t else Row([])
        # flatten nested Row to reduce braces later
        flat: List[Node] = []
        for it in items:
            if isinstance(it, Row):
                flat.extend(it.items)
            else:
                flat.append(it)
        return Row(flat)

    if tag == "mi":
        return Mi(_text(el))

    if tag == "mn":
        return Mn(_text(el))

    if tag == "mo":
        return Mo(_text(el))

    if tag == "mtext":
        return MText(_text(el))

    if tag == "mfrac":
        kids = _children(el)
        if len(kids) != 2:
            # weird malformed case -> row
            return Row([_convert(c) for c in kids])
        return Frac(_convert(kids[0]), _convert(kids[1]))

    if tag == "msqrt":
        kids = _children(el)
        if len(kids) == 1:
            return Sqrt(_convert(kids[0]))
        return Sqrt(Row([_convert(c) for c in kids]))

    if tag == "msup":
        kids = _children(el)
        if len(kids) != 2:
            return Row([_convert(c) for c in kids])
        return Sup(_convert(kids[0]), _convert(kids[1]))

    if tag == "msub":
        kids = _children(el)
        if len(kids) != 2:
            return Row([_convert(c) for c in kids])
        return Sub(_convert(kids[0]), _convert(kids[1]))

    if tag == "msubsup":
        kids = _children(el)
        if len(kids) != 3:
            return Row([_convert(c) for c in kids])
        return SubSup(_convert(kids[0]), _convert(kids[1]), _convert(kids[2]))

    if tag == "mfenced":
        # attributes can hold open/close, default is parentheses
        open_ch = el.attrib.get("open", "(")
        close_ch = el.attrib.get("close", ")")
        kids = _children(el)
        if len(kids) == 1:
            body = _convert(kids[0])
        else:
            body = Row([_convert(c) for c in kids])
        return Fenced(open=open_ch, body=body, close=close_ch)

    # fallback: treat unknown tags as row of children or text
    kids = _children(el)
    if kids:
        return Row([_convert(c) for c in kids])
    t = _text(el)
    return MText(t)