"""Narrow, discovery-informed public-property adapters, not a general semantic verifier."""
import ast
import re

from experiments.audit_public_examples import extract_examples


ROOT_SUPPORT = """
def _audit_public_root(value, coefficients):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    terms = [coefficient * value**index for index, coefficient in enumerate(coefficients)]
    if not all(math.isfinite(term) for term in terms):
        return False
    residual = abs(math.fsum(terms))
    scale = math.fsum(abs(term) for term in terms)
    return residual <= 1e-6 + 1e-9 * scale
"""


def polynomial_helper_matches(function):
    """Accept only the explicitly provided coefficient-order polynomial implementation."""
    if len(function.args.args) != 2:
        return False
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    if len(returns) != 1:
        return False
    value = returns[0].value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "sum" and len(value.args) == 1):
        return False
    terms = value.args[0]
    if not isinstance(terms, ast.ListComp) or len(terms.generators) != 1:
        return False
    target = terms.generators[0].target
    if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
            and all(isinstance(item, ast.Name) for item in target.elts)):
        return False
    index, coefficient = [item.id for item in target.elts]
    coefficients, point = [item.arg for item in function.args.args]
    template = (
        f"sum([{coefficient} * math.pow({point}, {index}) "
        f"for {index}, {coefficient} in enumerate({coefficients})])"
    )
    return ast.dump(value) == ast.dump(ast.parse(template, mode="eval").body)


def root_property(problem, extracted):
    try:
        tree = ast.parse(problem)
    except SyntaxError:
        return None
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, function in functions.items():
        doc = ast.get_docstring(function) or ""
        helper_match = re.search(r"find\s+x\s+such that\s+(\w+)\(x\)\s*=\s*0", doc)
        if not helper_match or len(function.args.args) != 1:
            continue
        helper = functions.get(helper_match.group(1))
        if helper is None or not polynomial_helper_matches(helper):
            continue
        checks = []
        for literal in extracted["checks"]:
            call = ast.parse(literal["expression"], mode="eval").body
            if isinstance(call.func, ast.Name) and call.func.id == "round":
                call = call.args[0]
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == name and len(call.args) == 1 and not call.keywords):
                continue
            coefficients = ast.literal_eval(call.args[0])
            if not (isinstance(coefficients, list) and coefficients
                    and all(type(value) in (int, float) for value in coefficients)):
                continue
            checks.append({
                "expression": f"_audit_public_root({ast.unparse(call)}, {coefficients!r})",
                "expected": True, "source_line": literal["source_line"],
            })
        if checks and len(checks) == len(extracted["checks"]):
            return {
                "mode": "polynomial_root_residual", "checks": checks,
                "support_code": ROOT_SUPPORT,
                "absolute_tolerance": 1e-6, "relative_tolerance": 1e-9,
                "limitation": "Finite public example inputs only; floating residual test, "
                              "not a proof or hidden-test correctness certificate.",
            }
    return None


def sort_conflicts(problem, extracted):
    """Detect only explicit non-negative popcount/decimal-order specification conflicts."""
    text = " ".join(problem.lower().split())
    markers = ("non-negative integers", "number of ones", "binary representation",
               "ascending order", "decimal value")
    if not all(marker in text for marker in markers):
        return []
    conflicts = []
    for check in extracted["checks"]:
        call = ast.parse(check["expression"], mode="eval").body
        if not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
            continue
        values = ast.literal_eval(call.args[0])
        if not isinstance(values, list) or not all(type(value) is int for value in values):
            continue
        if any(value < 0 for value in values):
            conflicts.append({"source_line": check["source_line"],
                              "reason": "example_outside_nonnegative_domain"})
            continue
        rule_result = sorted(values, key=lambda value: (value.bit_count(), value))
        if check["expected"] != rule_result:
            conflicts.append({"source_line": check["source_line"],
                              "reason": "example_disagrees_with_popcount_decimal_order",
                              "literal_expected": check["expected"], "rule_result": rule_result})
    return conflicts


def semantic_contract(problem):
    """Use only public prompt features. Unrecognized tasks retain literal checks or abstain."""
    extracted = extract_examples(problem)
    conflicts = sort_conflicts(problem, extracted)
    if conflicts:
        return {
            "mode": "specification_conflict_abstention", "checks": [],
            "support_code": "", "conflicts": conflicts,
        }
    root = root_property(problem, extracted)
    if root:
        return root
    return {
        "mode": "literal_examples" if extracted["checks"] else "unsupported",
        "checks": extracted["checks"], "support_code": "",
        "rejected_examples": extracted["rejected_examples"],
        "conflicting_expressions": extracted["conflicting_expressions"],
    }


def property_output(output, support_code):
    if not support_code:
        return output
    code = output.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Install the authored checker after candidate code so candidates cannot shadow its name.
    return f"exec({code!r}, globals())\n{support_code}"
