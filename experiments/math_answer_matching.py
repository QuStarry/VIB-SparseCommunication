"""Conservative whole-answer MATH matching, with no last-number fallback or Python eval."""
from __future__ import annotations

import ast
from functools import lru_cache
import re

import sympy as sp

VERSION = "whole_answer_v1"


def group(text, start):
    if start >= len(text):
        raise ValueError("missing TeX argument")
    if text[start] != "{":
        return text[start], start + 1
    depth = 1
    for index in range(start + 1, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if not depth:
            return text[start + 1:index], index + 1
    raise ValueError("unbalanced TeX braces")


def extract_answer(text):
    """Prefer the final balanced box; otherwise require a whole answer or explicit label."""
    positions = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if positions:
        return group(text, positions[-1].end() - 1)[0].strip()
    labels = list(re.finditer(
        r"(?:\u6700\u7ec8\u7b54\u6848|\u7b54\u6848|final answer|the answer|answer)"
        r"\s*(?:is\s*)?[:\uff1a=]\s*", text, flags=re.I
    ))
    return text[labels[-1].end():].strip() if labels else text.strip()


def convert_tex(text):
    result = []
    index = 0
    while index < len(text):
        if text.startswith("\\frac", index):
            index += 5
            while index < len(text) and text[index].isspace():
                index += 1
            numerator, index = group(text, index)
            while index < len(text) and text[index].isspace():
                index += 1
            denominator, index = group(text, index)
            result.append(f"(({convert_tex(numerator)})/({convert_tex(denominator)}))")
        elif text.startswith("\\sqrt", index):
            index += 5
            degree = "2"
            if index < len(text) and text[index] == "[":
                closing = text.index("]", index)
                degree = text[index + 1:closing]
                if not degree.isdigit() or not 1 <= int(degree) <= 100:
                    raise ValueError("unsupported root degree")
                index = closing + 1
            while index < len(text) and text[index].isspace():
                index += 1
            radicand, index = group(text, index)
            result.append(f"(({convert_tex(radicand)})^(1/{degree}))")
        else:
            result.append(text[index])
            index += 1
    return "".join(result)


def normalize(text):
    text = extract_answer(text)
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$", "$")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
    currency = text.startswith((r"\$", "$"))
    if currency:
        text = text.removeprefix(r"\$").removeprefix("$")
    for command in ("\\dfrac", "\\tfrac", "\\cfrac"):
        text = text.replace(command, "\\frac")
    for command in ("\\left", "\\right", "\\displaystyle", "\\!", "\\,", "\\;", "\\quad"):
        text = text.replace(command, "")
    substitutions = {
        "\u2212": "-", "\u221e": "oo", "\\infty": "oo", "\\pi": "pi",
        "\u03c0": "pi", "\u222a": "|", "\\cup": "|", "\\cdot": "*", "\\times": "*",
        "\u00d7": "*", "\u00b2": "^2", "\u00b3": "^3",
        "\\{": "{", "\\}": "}", "\\%": "%", "^\\circ": "",
    }
    for source, target in substitutions.items():
        text = text.replace(source, target)
    text = re.sub(r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(
        r"\s*(?:students|dollars|seconds|meters|centimeters|degrees|cm|feet|inches)\s*$",
        "", text, flags=re.I
    )
    if currency and re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    label = None
    assignment = re.fullmatch(r"([a-z])\s*(?:=|\\in)\s*(.+)", text)
    if assignment:
        label, text = assignment.groups()
    text = convert_tex(text)
    return label, text.strip()


def split_top_level(text, separator):
    depth = 0
    pieces = []
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    return pieces


def expression(text):
    text = text.replace("{", "(").replace("}", ")").replace("^", "**")
    if any(word not in ("pi", "oo", "sqrt") for word in re.findall(r"[a-zA-Z]{3,}", text)):
        raise ValueError("text answers are not implicit symbolic products")
    tokens = re.findall(r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+|sqrt|pi|oo|[a-z]|\*\*|[()+*/-]", text)
    if re.sub(r"\s+", "", text) != "".join(tokens) or len(tokens) > 128:
        raise ValueError("unsupported expression syntax")
    joined = []
    for token in tokens:
        if joined:
            previous = joined[-1]
            previous_atom = previous == ")" or previous in ("pi", "oo") or previous[-1:].isalnum()
            next_atom = token == "(" or token[0].isalnum() or token.startswith(".")
            if previous_atom and next_atom and not (previous == "sqrt" and token == "("):
                joined.append("*")
        joined.append(token)
    source = "".join(joined)
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 192:
        raise ValueError("expression too large")

    def convert(node):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords):
            return sp.sqrt(convert(node.args[0]))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(source, node)
            if len(literal) > 60:
                raise ValueError("numeric literal too large")
            exponent = re.search(r"[eE]([-+]?\d+)$", literal)
            if exponent and abs(int(exponent[1])) > 100:
                raise ValueError("numeric exponent too large")
            return sp.Rational(literal)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return sp.pi
            if node.id == "oo":
                return sp.oo
            if node.id == "i":
                return sp.I
            if re.fullmatch("[a-z]", node.id):
                return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Rational and abs(right) <= 100:
                if left.free_symbols and abs(right) > 20:
                    raise ValueError("symbolic degree too large")
                return left ** right
        raise ValueError("unsupported expression node")

    value = convert(tree.body)
    if value.has(sp.zoo, sp.nan):
        raise ValueError("undefined expression")
    return value


@lru_cache(maxsize=20000)
def canonical(text):
    if len(text) > 2048:
        raise ValueError("answer too long")
    # Mixed interval endpoints are valid, but missing or extra delimiters are not.
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if depth < 0:
            raise ValueError("unbalanced answer delimiters")
    if depth:
        raise ValueError("unbalanced answer delimiters")
    unions = split_top_level(text, "|")
    if len(unions) > 1:
        return ("union", tuple(canonical(item.strip()) for item in unions))
    if len(text) > 1 and text[0] in "([{" and text[-1] in ")]}":
        inner = split_top_level(text[1:-1], ",")
        if len(inner) > 1 or text[0] + text[-1] == "{}":
            return (text[0] + text[-1], tuple(canonical(item.strip()) for item in inner))
    parts = split_top_level(text, ",")
    if len(parts) > 1:
        return ("ordered_list", tuple(canonical(item.strip()) for item in parts))
    try:
        return ("expression", expression(text))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
        return ("opaque", re.sub(r"\s+", "", text))


def equivalent(left, right):
    if left[0] != right[0]:
        return False
    if left[0] == "expression":
        return left[1] == right[1] or sp.cancel(left[1] - right[1]) == 0
    if left[0] == "opaque":
        return left[1] == right[1]
    if len(left[1]) != len(right[1]):
        return False
    if left[0] in ("{}", "union"):
        remaining = list(right[1])
        for item in left[1]:
            match = next((index for index, candidate in enumerate(remaining)
                          if equivalent(item, candidate)), None)
            if match is None:
                return False
            remaining.pop(match)
        return True
    return all(equivalent(a, b) for a, b in zip(left[1], right[1]))


@lru_cache(maxsize=30000)
def score(prediction, reference):
    """Return a decision and extraction diagnostics; unresolved mismatches stay false."""
    try:
        pred_label, pred = normalize(prediction)
        ref_label, ref = normalize(reference)
        if not pred or not ref:
            return {"passed": False, "method": VERSION, "status": "empty_answer"}
        pred_value, ref_value = canonical(pred), canonical(ref)
        matched = not (pred_label and ref_label and pred_label != ref_label)
        matched = matched and equivalent(pred_value, ref_value)
        if not matched and not (pred_label and ref_label and pred_label != ref_label) and ref_value[0] == "expression":
            if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", pred):
                matched = equivalent(canonical(pred.replace(",", "")), ref_value)
        return {
            "passed": bool(matched), "prediction_answer": pred, "reference_answer": ref,
            "prediction_kind": pred_value[0], "reference_kind": ref_value[0],
            "method": VERSION,
        }
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, NotImplementedError):
        return {"passed": False, "method": VERSION, "status": "unsupported_or_malformed"}
