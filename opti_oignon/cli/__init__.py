#!/usr/bin/env python3
"""
Opti-Oignon CLI companion package.

Provides the ``oo`` command-line tool for interacting with a running
Opti-Oignon backend from the terminal.  Most commands communicate only
through the HTTP/WebSocket API.  Three run in this process: ``oo chat``,
whose session drives the executor, the conversation store, the onion
memory and the skill registry here, with inference through the registry;
``oo core``, which serves or probes the core daemon; and ``oo garden``,
which looks after the componion through its service.  Each imports the
package inside the command, never at module load.
"""
