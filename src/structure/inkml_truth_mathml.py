# src/structure/inkml_truth_mathml.py
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional

def _local(tag: str) -> str:
    """Strip namespace: '{ns}mrow' -> 'mrow'."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def extract_truth_mathml_root(inkml_path: str | Path) -> Optional[ET.Element]:
    """
    Returns the <math> element inside annotationXML truth Content-MathML,
    or None if not found.
    """
    inkml_path = Path(inkml_path)
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()

    # find annotationXML nodes regardless of namespace
    for annxml in root.iter():
        if _local(annxml.tag) != "annotationXML":
            continue
        t = annxml.attrib.get("type", "")
        enc = annxml.attrib.get("encoding", "")
        if t == "truth" and "MathML" in enc:
            # find first <math> descendant
            for node in annxml.iter():
                if _local(node.tag) == "math":
                    return node
    return None

def extract_truth_latex_string(inkml_path: str | Path) -> Optional[str]:
    """
    Reads <annotation type="truth"> ... latex ... </annotation>
    """
    inkml_path = Path(inkml_path)
    tree = ET.parse(str(inkml_path))
    root = tree.getroot()

    for ann in root.iter():
        if _local(ann.tag) != "annotation":
            continue
        if ann.attrib.get("type", "") == "truth":
            if ann.text is None:
                return None
            return ann.text.strip()
    return None