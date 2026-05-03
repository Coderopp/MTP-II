#!/usr/bin/env bash
# Bundles data/results/ into a tarball and uploads to catbox.moe.
# Prints the download URL on the last line — read that URL off the screen
# and pass it to the laptop side, which will wget + extract it.
#
# Used to ferry experiment outputs out of the iDRAC server when:
#   - SSH/rsync sessions are blocked at the firewall, AND
#   - Paste is broken on the iDRAC console (so PATs / SSH keys can't be typed)
#
# catbox.moe accepts files up to 200 MB.  Files older than ~30 days may
# be garbage-collected, so download promptly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/data/results"
TARBALL="/tmp/mtp_results_$(date +%Y%m%d_%H%M%S).tar.gz"

if [[ ! -d "$RESULTS" ]]; then
  echo "ERROR: $RESULTS does not exist" >&2
  exit 1
fi

echo "Bundling $RESULTS -> $TARBALL"
tar czf "$TARBALL" -C "$REPO_ROOT/data" results
ls -lh "$TARBALL"

echo "Uploading to catbox.moe (up to 200 MB) ..."
URL=$(curl -s --max-time 120 \
  -F "reqtype=fileupload" \
  -F "fileToUpload=@$TARBALL" \
  https://catbox.moe/user/api.php)

if [[ -z "$URL" || "$URL" != https://* ]]; then
  echo "ERROR: upload failed; raw response: $URL" >&2
  exit 1
fi

echo
echo "============================================================"
echo "DOWNLOAD URL (read this off the screen and tell the laptop):"
echo "$URL"
echo "============================================================"
