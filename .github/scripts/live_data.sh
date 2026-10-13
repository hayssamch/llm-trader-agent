#!/usr/bin/env bash
# Persist the daily-signal state (state.json, signals_log.csv, paper_trades.csv) on a
# dedicated branch. It is a separate branch ON PURPOSE: Streamlit Cloud redeploys the
# dashboard on every push to main, and a daily log commit must never restart the app
# (it would interrupt anyone mid-backtest).
#
#   live_data.sh pull   # fetch the branch (or start a fresh one on the first run)
#   live_data.sh push   # commit and push whatever the run changed
#
# Auth: LIVE_DATA_REMOTE if set (used by the tests), otherwise built from GH_TOKEN and
# GITHUB_REPOSITORY. The URL contains a token, so it is never echoed.
set -euo pipefail

DIR="${LIVE_DATA_DIR:-live_data}"
BRANCH="${LIVE_DATA_BRANCH:-live-data}"
REMOTE="${LIVE_DATA_REMOTE:-https://x-access-token:${GH_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"

case "${1:-}" in
  pull)
    rm -rf "$DIR"
    rc=0
    git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE" "$DIR"
      echo "Loaded saved state from branch '$BRANCH'."
    elif [ "$rc" -eq 2 ]; then
      # exit code 2 = reachable, but the branch does not exist yet (first run)
      mkdir -p "$DIR"
      git -C "$DIR" init --quiet -b "$BRANCH"
      git -C "$DIR" remote add origin "$REMOTE"
      echo "No '$BRANCH' branch yet — starting fresh."
    else
      # Any other failure (network, auth): stop. Starting "fresh" here could overwrite
      # real history with an empty state.
      echo "::error::Could not reach the repository to load saved state (git exit $rc)." >&2
      exit "$rc"
    fi
    ;;
  push)
    cd "$DIR"
    git config user.name  "daily-signals[bot]"
    git config user.email "daily-signals@users.noreply.github.com"
    git add -A
    if git diff --cached --quiet; then
      echo "Nothing new to save."
      exit 0
    fi
    git commit --quiet -m "Daily signals $(date -u +%F)"
    git push --quiet origin "$BRANCH"
    echo "Saved state to branch '$BRANCH'."
    ;;
  *)
    echo "usage: live_data.sh pull|push" >&2
    exit 64
    ;;
esac
