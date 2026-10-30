#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
mise run build
APP="build/Build/Products/Release/OverhearSubs.app"
RES="$APP/Contents/Resources/backend"
mkdir -p "$RES"
cp "$REPO_ROOT"/app.py "$REPO_ROOT"/pipeline.py "$REPO_ROOT"/backends.py "$RES/"
cp -R "$REPO_ROOT"/static "$RES/"
cp -R "$REPO_ROOT"/.venv "$RES/.venv"
echo "Bundled backend into $APP"
# create dmg
hdiutil create -volname OverhearSubs -srcfolder "$APP" -ov -format UDZO OverhearSubs.dmg
echo "Created OverhearSubs.dmg"