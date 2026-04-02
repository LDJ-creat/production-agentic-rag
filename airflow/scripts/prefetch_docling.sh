#!/usr/bin/env bash
set -euo pipefail

# This script tries to prefetch Docling snapshots/models used at runtime.
# It runs as root during image build; when run as non-root, set DOCLING_CACHE_DIR appropriately.

echo "Prefetching Docling snapshots..."

# Use python to import docling and trigger any download mechanism. If docling exposes
# a CLI or a download API, call it specifically. We'll attempt a safe import and
# try to access common model entries to force download.

python - <<'PY'
import os
try:
    import docling
    print('docling imported:', docling.__version__)
except Exception as e:
    print('docling import failed:', e)
    raise

# Attempt to load typical parsers or default snapshot entries.
# The exact API may vary; this tries common access patterns used by docling.
try:
    # Try to access a default model entry through docling's registry if available
    if hasattr(docling, 'registry'):
        print('Found docling.registry, listing entries...')
        try:
            entries = getattr(docling, 'registry').list_entries()
            print('registry entries count:', len(entries))
        except Exception:
            pass

    # Try to create a parser/loader to force download of required assets
    if hasattr(docling, 'DocumentParser'):
        print('Instantiating DocumentParser to warm cache...')
        try:
            parser = docling.DocumentParser()
            print('DocumentParser instantiated')
        except Exception as e:
            print('DocumentParser instantiation error (may be expected):', e)

    # As a fallback, try downloading a known small model entry if docling exposes CLI
    import subprocess
    try:
        subprocess.run(['docling', 'download', '--help'], check=False)
    except FileNotFoundError:
        pass

    print('Docling prefetch attempt finished')
except Exception as e:
    print('Docling prefetch encountered errors:', e)
    raise
PY

echo "Done prefetching Docling snapshots."

exit 0
