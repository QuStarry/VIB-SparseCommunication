import random

from experiments.audit_inverse_contracts import (
    evaluate_inverse, extract_inverse_contract, generate_strings, validate_public_encoder,
)


PUBLIC_REVERSE = '''
def supplied_transform(s: str):
    return s[::-1]
def inverse_transform(s: str):
    """takes as input string encoded with supplied_transform function. Returns decoded string."""
'''


def test_inverse_contract_uses_public_reference_not_function_name():
    contract = extract_inverse_contract(PUBLIC_REVERSE)
    assert contract["decoder"] == "inverse_transform"
    renamed = PUBLIC_REVERSE.replace("supplied_transform", "other_encoder").replace(
        "inverse_transform", "other_decoder"
    )
    assert extract_inverse_contract(renamed)["decoder"] == "other_decoder"


def test_unprovided_or_unsupported_encoder_is_not_adapted():
    missing = PUBLIC_REVERSE.replace("supplied_transform function", "absent function")
    unsafe = PUBLIC_REVERSE.replace("return s[::-1]", "return __import__('os').getcwd()")
    assert extract_inverse_contract(missing) is None
    assert extract_inverse_contract(unsafe) is None


def test_public_noninjective_encoder_causes_abstention():
    contract = extract_inverse_contract(PUBLIC_REVERSE.replace("return s[::-1]", "return ''"))
    guard = validate_public_encoder(contract, ["a", "b"])
    assert not guard["valid"]
    assert guard["reason"] == "encoder_collision_on_probe_domain"


def test_synthetic_correct_inverse_passes_generated_probes():
    contract = extract_inverse_contract(PUBLIC_REVERSE)
    output = PUBLIC_REVERSE + "\n    return s[::-1]"
    report = evaluate_inverse(output, contract, generate_strings())
    assert report["all_roundtrips_satisfied"]
    assert report["satisfied_probes"] == 64
    assert report["supplied_encoder_symbol_present"]


def test_roundtrip_failure_is_separate_from_exception():
    contract = extract_inverse_contract(PUBLIC_REVERSE)
    wrong = evaluate_inverse("def inverse_transform(s): return 'wrong'", contract, ["ab"])
    error = evaluate_inverse("def inverse_transform(s): raise ValueError()", contract, ["ab"])
    assert wrong["property_violations"] == 1
    assert wrong["probe_execution_errors"] == 0
    assert error["property_violations"] == 0
    assert error["probe_execution_errors"] == 1


def test_missing_supplied_helper_does_not_imply_decoder_semantic_failure():
    contract = extract_inverse_contract(PUBLIC_REVERSE)
    report = evaluate_inverse("def inverse_transform(s): return s[::-1]", contract, ["abc"])
    assert report["all_roundtrips_satisfied"]
    assert not report["supplied_encoder_symbol_present"]


def test_generated_inputs_are_unique_lowercase_and_rng_independent():
    state = random.getstate()
    try:
        samples = generate_strings()
        assert len(samples) == len(set(samples)) == 64
        assert samples == generate_strings()
        assert "" in samples
        assert {0, 1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33, 64, 65} <= set(map(len, samples))
        assert all(set(sample) <= set("abcdefghijklmnopqrstuvwxyz") for sample in samples)
        assert random.getstate() == state
    finally:
        random.setstate(state)


def test_ambiguous_multiple_inverse_functions_abstain():
    another = '''
def another(s):
    """takes as input string encoded with supplied_transform function. Returns decoded string."""
'''
    assert extract_inverse_contract(PUBLIC_REVERSE + another) is None
