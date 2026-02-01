from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET


MATHML_NS = "http://www.w3.org/1998/Math/MathML"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def T(tag: str) -> str:
    return f"{{{MATHML_NS}}}{tag}"


def xml_id(elem: ET.Element) -> Optional[str]:
    return elem.attrib.get(f"{{{XML_NS}}}id") or elem.attrib.get("id")


@dataclass
class Edge:
    src: str
    rel: str
    dst: str


@dataclass
class VirtualNode:
    node_id: str
    kind: str  # e.g. "FRAC"


def _is_token(elem: ET.Element) -> bool:
    return elem.tag in {T("mi"), T("mn"), T("mo"), T("mtext")}


def _collect_token_ids(elem: ET.Element) -> List[str]:
    """
    Collect xml:id of all token leaves under elem, in document order.
    """
    out: List[str] = []
    for n in elem.iter():
        if _is_token(n):
            i = xml_id(n)
            if i:
                out.append(i)
    return out


def _add_next_edges(edges: List[Edge], token_ids: List[str]) -> None:
    for a, b in zip(token_ids, token_ids[1:]):
        edges.append(Edge(a, "NEXT", b))


def extract_edges_from_mathml(math_elem: ET.Element) -> Tuple[List[str], List[Edge], List[VirtualNode]]:
    """
    Returns:
      token_ids: list of all token xml:ids found (in doc order)
      edges: structural + reading-order edges
      virtual_nodes: virtual operator nodes (e.g., FRAC nodes) used for clean supervision
    """
    edges: List[Edge] = []
    virtual_nodes: List[VirtualNode] = []

    token_ids = _collect_token_ids(math_elem)

    frac_counter = 0

    def new_virtual(kind: str) -> str:
        nonlocal frac_counter
        if kind == "FRAC":
            frac_counter += 1
            vid = f"FRAC@{frac_counter}"
            virtual_nodes.append(VirtualNode(vid, "FRAC"))
            return vid
        # extend later if you add ROOT@k, SQRT@k, etc.
        frac_counter += 1
        vid = f"{kind}@{frac_counter}"
        virtual_nodes.append(VirtualNode(vid, kind))
        return vid

    def walk(node: ET.Element):
        tag = node.tag

        # -------------------------
        # Reading order inside mrow
        # -------------------------
        if tag == T("mrow"):
            children = list(node)
            child_token_lists = [_collect_token_ids(ch) for ch in children]
            for a, b in zip(child_token_lists, child_token_lists[1:]):
                if a and b:
                    edges.append(Edge(a[-1], "NEXT", b[0]))

        # -------------------------
        # msup / msub / msubsup
        # -------------------------
        if tag == T("msup") and len(node) == 2:
            base_ids = _collect_token_ids(node[0])
            exp_ids = _collect_token_ids(node[1])
            if base_ids and exp_ids:
                edges.append(Edge(base_ids[-1], "SUP", exp_ids[0]))

        if tag == T("msub") and len(node) == 2:
            base_ids = _collect_token_ids(node[0])
            sub_ids = _collect_token_ids(node[1])
            if base_ids and sub_ids:
                edges.append(Edge(base_ids[-1], "SUB", sub_ids[0]))

        if tag == T("msubsup") and len(node) == 3:
            base_ids = _collect_token_ids(node[0])
            sub_ids = _collect_token_ids(node[1])
            sup_ids = _collect_token_ids(node[2])
            if base_ids and sub_ids:
                edges.append(Edge(base_ids[-1], "SUB", sub_ids[0]))
            if base_ids and sup_ids:
                edges.append(Edge(base_ids[-1], "SUP", sup_ids[0]))

        # -------------------------
        # mfrac: USE A VIRTUAL NODE
        # FRAC@k -> numerator_first (NUM)
        # FRAC@k -> denominator_first (DEN)
        # -------------------------
        if tag == T("mfrac") and len(node) == 2:
            num_ids = _collect_token_ids(node[0])
            den_ids = _collect_token_ids(node[1])
            if num_ids and den_ids:
                fid = new_virtual("FRAC")
                edges.append(Edge(fid, "FRAC_NUM", num_ids[0]))
                edges.append(Edge(fid, "FRAC_DEN", den_ids[0]))

        # -------------------------
        # munderover (big ops)
        # -------------------------
        if tag == T("munderover") and len(node) == 3:
            op_ids = _collect_token_ids(node[0])
            under_ids = _collect_token_ids(node[1])
            over_ids = _collect_token_ids(node[2])
            if op_ids and under_ids:
                edges.append(Edge(op_ids[0], "UNDER", under_ids[0]))
            if op_ids and over_ids:
                edges.append(Edge(op_ids[0], "OVER", over_ids[0]))

        # recurse
        for ch in list(node):
            walk(ch)

    walk(math_elem)
    return token_ids, edges, virtual_nodes


def find_content_mathml_root(inkml_root: ET.Element) -> Optional[ET.Element]:
    """
    Locate <annotationXML type="truth" encoding="Content-MathML"> ... <math> ... </math>
    and return the <math> element.
    """
    for ax in inkml_root.iter():
        if ax.tag.endswith("annotationXML"):
            if ax.attrib.get("type") == "truth" and ax.attrib.get("encoding") == "Content-MathML":
                for child in list(ax):
                    if child.tag == T("math"):
                        return child
    return None