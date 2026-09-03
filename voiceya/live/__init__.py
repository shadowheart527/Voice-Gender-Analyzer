"""Sentence-live analysis: a WebSocket session that analyses speech pause by pause.

See docs/plans/sentence_live_mode.md.  This package is served by its own process
(``python -m voiceya.live``) so the API process stays free of TensorFlow and the
taskiq queue is not involved in stateful sessions.
"""
