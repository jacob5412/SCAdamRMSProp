import json
import os

import pytest


def load_json(file_path):
    """
    Load a JSON file if it exists, otherwise return an empty dictionary.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}


def save_json(data, file_path):
    """
    Save a dictionary to a JSON file, creating directories if necessary.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to access test reports for each phase of a test: setup, call, teardown.
    """
    # Execute all other hooks to get the report object
    outcome = yield
    rep = outcome.get_result()

    # Attach the report to the test item for later access
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(scope="function", autouse=True)
def log_test_result(request):
    """
    Log test results to a JSON file.
    Each test is marked as passed or failed in `tests/test_results.json`.
    """
    yield  # Run the test

    # Ensure the test result is available and handle missing attributes gracefully
    test_results_file = "tests/test_results.json"
    test_results = load_json(test_results_file)

    test_name = request.node.name
    rep_call = getattr(request.node, "rep_call", None)

    # Only log results if `rep_call` exists
    if rep_call:
        test_results[test_name] = rep_call.passed
    else:
        # If `rep_call` is missing, mark the test as failed
        test_results[test_name] = False

    save_json(test_results, test_results_file)


@pytest.fixture(scope="session", autouse=True)
def generate_summary(request):
    """
    Generate a summary of test results at the end of the test session.
    This writes to `tests/test_summary.json` and logs the results to the console.
    """
    yield  # Run all tests
    test_results_file = "tests/test_results.json"
    summary_file = "tests/test_summary.json"
    test_results = load_json(test_results_file)

    summary = {
        "total_tests": len(test_results),
        "passed_tests": sum(test_results.values()),
        "failed_tests": len(test_results) - sum(test_results.values()),
        "details": test_results,
    }

    save_json(summary, summary_file)

    # Print the summary to the console
    print("\n======================")
    print("Test Summary")
    print("======================")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed tests: {summary['passed_tests']}")
    print(f"Failed tests: {summary['failed_tests']}")
    print("======================")
    if summary["failed_tests"] > 0:
        print("Failed test cases:")
        for test, passed in test_results.items():
            if not passed:
                print(f"- {test}")
    print("======================")
