#!/usr/bin/env python3
"""What a contract leaves behind: nothing it did not find.

Contracts that load a module in isolation work on state the whole test
process shares -- the module cache, and the transport some of them replace.
Left changed, that state reaches every suite that runs later, and a later
contract then passes or fails for a reason it never established: the facade
contract did, red or green with the order of the suites before it, because a
stand-in package was left in the cache.

Each of these now fails the contract that leaves it, at its teardown:

  * a project module with no file -- a stand-in -- that the contract did not
    find in the module cache, the leftover behind the facade contract;
  * ``urllib.request.urlopen`` left as anything but what the contract found,
    which five suites did;
  * a neutralised project entry (``None``) the contract did not find. None
    was found, but one would make every later import of that name fail.

The state is read before any fixture of the contract is set up and checked
after every one of them is torn down, so a suite that closes its own window
in a fixture is not charged for it. A real module a contract imports for the
first time is not a leftover. The check reads and never writes: it cannot
turn a red contract green.

Local-only (the public distribution ships no tests).
"""

import sys
import urllib.request

import pytest

_PACKAGE = "opti_oignon"
_FOUND = pytest.StashKey()


def _project_entries():
    """The project's entries in the module cache, by name."""
    return {
        name: module
        for name, module in sys.modules.items()
        if name == _PACKAGE or name.startswith(_PACKAGE + ".")
    }


def left_behind(found, now):
    """Names now held by a stand-in or ``None`` that the contract did not find there."""
    return sorted(
        name
        for name, module in now.items()
        if (name not in found or found[name] is not module)
        and (module is None or not getattr(module, "__file__", None))
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    item.stash[_FOUND] = (_project_entries(), urllib.request.urlopen)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    result = yield
    found = item.stash.get(_FOUND, None)
    if found is not None:
        entries, transport = found
        problems = []
        left = left_behind(entries, _project_entries())
        if left:
            problems.append(f"project modules left as stand-ins or neutralised: {left[:6]}")
        if urllib.request.urlopen is not transport:
            problems.append("urllib.request.urlopen is left replaced")
        if problems:
            raise AssertionError("; ".join(problems))
    return result
