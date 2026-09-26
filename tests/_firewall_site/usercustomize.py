"""The data firewall of a test session, installed in each Python child it starts.

``DataFirewall.cover_children`` in ``tests/_data_firewall.py`` sets two
variables -- the roots of the covered trees and their mirrors, each a list
joined with ``os.pathsep`` -- and puts this directory first on
``PYTHONPATH``, then the tree's root. The interpreter's site hook imports
``usercustomize`` at startup, before any code of the child's own, so a
child that inherits that environment runs this module first, and it
installs one firewall per root, lazily: the firewall's source is executed
under a private name, never entered in the module cache, so nothing the
interpreter had not loaded is loaded, ``sys.path`` is not touched, and a
test session started inside the child imports its own copy of the
firewall. ``COVERED`` then names the roots covered here, which is what the
child ``cover_children`` starts to check reads back.

Without the variables this does nothing. With them, a child that cannot be
covered does not run: it names the reason on its standard error and exits
with status 70, because running uncovered is the failure this exists to
prevent, and a warning a child prints and goes on from is not seen.

Local-only (the public distribution ships no tests).
"""

import os

# The status of a child that refuses to run uncovered (EX_SOFTWARE).
REFUSED = 70


def _refuse(reason):
    """Say why on the standard error, then end this process at once."""
    try:
        os.write(2, f"data firewall, child process {os.getpid()}: {reason}; it refuses to run uncovered\n"
                 .encode("utf-8", "replace"))
    finally:
        os._exit(REFUSED)


def _cover():
    roots = os.environ.get("OO_TEST_FIREWALL_ROOT", "")
    mirrors = os.environ.get("OO_TEST_FIREWALL_MIRROR", "")
    if not roots and not mirrors:
        return ()
    roots, mirrors = roots.split(os.pathsep), mirrors.split(os.pathsep)
    if len(roots) != len(mirrors) or not all(os.path.isabs(path) for path in roots + mirrors):
        _refuse(f"the roots {roots!r} and the mirrors {mirrors!r} do not pair as absolute paths")
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_data_firewall.py")
    namespace = {"__name__": "_data_firewall_in_child", "__file__": source}
    try:
        with open(source, encoding="utf-8") as handle:
            code = compile(handle.read(), source, "exec")
        exec(code, namespace)
        firewalls = namespace["cover_this_child"](roots, mirrors)
    except BaseException as error:  # noqa: BLE001 - whatever stops the install stops the child
        _refuse(f"the firewall of {source} could not be installed ({type(error).__name__}: {error})")
    return tuple(firewall.root for firewall in firewalls)


# The roots whose firewall is installed in this process.
COVERED = _cover()
