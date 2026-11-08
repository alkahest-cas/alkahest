#!/usr/bin/env bash
# Download a source tarball or tool, surviving a flaky host.
#
#   fetch.sh OUTPUT SHA256 URL [MIRROR_URL...]
#
# SHA256 may be `-` to skip verification. Each URL is tried with curl's own
# retries, in order, then the whole list again: on 2026-11-08 a GitHub incident
# failed release and CI jobs on 504s and on an HTML error page served with
# status 200, which is why a checksum mismatch is treated as one more failed
# attempt rather than the end of the job.
#
# `--retry-all-errors` needs curl >= 7.71. The manylinux_2_28 image (AlmaLinux 8)
# ships 7.61, where the flag is an "unknown option" error, so it is added only
# when curl has it; the outer loop covers what the older curl will not retry.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: fetch.sh OUTPUT SHA256|- URL [MIRROR_URL...]" >&2
  exit 2
fi
out=$1
sum=$2
shift 2

opts=(--fail --location --silent --show-error --connect-timeout 30 --max-time 900
      --retry 5 --retry-delay 15)
# Not `curl --help all | grep -q`: grep exits at the first match, curl dies of
# SIGPIPE, and under pipefail the pipeline "fails" when the flag is present.
curl_help=$(curl --help all 2>/dev/null || true)
case "$curl_help" in
  *--retry-all-errors*) opts+=(--retry-all-errors) ;;
esac

for round in 1 2 3; do
  for url in "$@"; do
    echo "fetch: $url (round $round)"
    rm -f "$out"
    if curl "${opts[@]}" -o "$out" "$url"; then
      if [ "$sum" = - ]; then
        exit 0
      fi
      if echo "$sum  $out" | sha256sum --check --status -; then
        echo "fetch: $out verified"
        exit 0
      fi
      echo "fetch: checksum mismatch for $url; got $(sha256sum "$out" | cut -d' ' -f1)" >&2
      head -c 200 "$out" | tr -cd '[:print:]\n' >&2 || true
      echo >&2
    fi
  done
  if [ "$round" != 3 ]; then
    sleep 30
  fi
done

echo "fetch: could not download $out from any of: $*" >&2
exit 1
