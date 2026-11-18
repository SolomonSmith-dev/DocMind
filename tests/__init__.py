import os

# The test suite runs with the built-in dev secrets; production boot refuses them.
os.environ.setdefault("DOCMIND_ALLOW_INSECURE_DEFAULTS", "true")
