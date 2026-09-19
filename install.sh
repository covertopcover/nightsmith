#!/bin/sh
# nightsmith installer — https://nightsmith.sh/install
#
# The promises this script is here to keep:
#
#   * It never calls sudo and never writes outside $HOME. That is what makes
#     "it never asks for your password" literally true, and it is what makes
#     `nightsmith remove` a promise the tool can keep.
#   * It ends by running the tool, so nobody ever meets "now restart your
#     terminal" — a dead stop for the audience this is built for.
#   * --dry-run lists every file it would touch and exits, because piping a
#     stranger's script to sh deserves an inspectable answer.
#
# Deliberately not here: installing the model runtime or downloading weights.
# This is ~5 MB and takes seconds; the model is gigabytes and asks first.
# Conflating the two is how people get ambushed by a multi-gigabyte download.

set -eu

# Releases live on GitHub; nightsmith.sh/install points at this script.
# Override either for testing.
REPO="${NIGHTSMITH_REPO:-covertopcover/nightsmith}"
VERSION="${NIGHTSMITH_VERSION:-latest}"

BIN_DIR="${NIGHTSMITH_BIN_DIR:-$HOME/.local/bin}"
BIN="$BIN_DIR/nightsmith"
ASSET="nightsmith-aarch64-apple-darwin"

DRY_RUN=0
NO_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --no-run)  NO_RUN=1 ;;
        --help|-h)
            echo "usage: install.sh [--dry-run] [--no-run]"
            echo "  --dry-run  list every file this would touch, then exit"
            echo "  --no-run   install, but do not start setup afterwards"
            exit 0 ;;
        *) echo "install.sh: unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '  %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$*"; }
die()  { printf '\n  \033[31m✗\033[0m %s\n\n' "$*" >&2; exit 1; }

# ── Is this a Mac this can work on? ─────────────────────────────────────────
# Refuse clearly rather than fall back. A CPU fallback
# would be too slow to function as a product while still generating the support
# load of one, and a paid fallback contradicts the free premise.

[ "$(uname -s)" = "Darwin" ] || die "nightsmith is macOS only — this is $(uname -s)."

arch="$(uname -m)"
if [ "$arch" != "arm64" ]; then
    # uname lies under Rosetta: an x86_64-built terminal on Apple Silicon
    # reports x86_64 for everything it spawns. Refusing here without this check
    # would turn away working Macs, and the user could never work out why.
    translated="$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)"
    if [ "$translated" = "1" ]; then
        die "This terminal is running under Rosetta, so it reports an Intel CPU.
    This Mac is Apple Silicon and nightsmith will work here.

    Right-click your terminal app in Applications → Get Info → untick
    \"Open using Rosetta\", reopen it, and run this again."
    fi
    die "This Mac has an Intel processor. MLX needs Apple Silicon (M1 or
    newer), so nightsmith can't run here.

    It'll work on any Mac from 2020 onward with an M-series chip."
fi

macos="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
ok "Apple Silicon · macOS $macos"

# ── Where things go ─────────────────────────────────────────────────────────
# ~/.local/bin, never /usr/local/bin, which is root:wheel and would need a
# password. Ollama's installer falls back to `sudo ln -sf`; not copying that is
# the concrete difference this product is making.

case "$SHELL" in
    */zsh)  RC="$HOME/.zshrc" ;;
    */bash) RC="$HOME/.bash_profile" ;;
    */fish) RC="$HOME/.config/fish/config.fish" ;;
    *)      RC="$HOME/.profile" ;;
esac

path_has_bin_dir() {
    case ":${PATH}:" in *":${BIN_DIR}:"*) return 0 ;; *) return 1 ;; esac
}
rc_has_bin_dir() {
    [ -f "$RC" ] && grep -q "$BIN_DIR" "$RC" 2>/dev/null
}

need_path_edit=0
if ! path_has_bin_dir && ! rc_has_bin_dir; then
    need_path_edit=1
fi

if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/$REPO/releases/latest/download/$ASSET"
    SUMS_URL="https://github.com/$REPO/releases/latest/download/SHA256SUMS"
else
    URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
    SUMS_URL="https://github.com/$REPO/releases/download/$VERSION/SHA256SUMS"
fi

if [ "$DRY_RUN" = "1" ]; then
    echo
    say "nightsmith --dry-run. Nothing below has happened."
    echo
    say "Would download:"
    say "    $URL"
    say "    $SUMS_URL   (checked before anything is installed)"
    echo
    say "Would write:"
    say "    $BIN"
    [ "$need_path_edit" = "1" ] \
        && say "    $RC          (one line adding $BIN_DIR to PATH)" \
        || say "    nothing else — $BIN_DIR is already on your PATH"
    echo
    say "Would not touch: /usr/local, /Applications, /Library, any system"
    say "file, your Python, your Homebrew, or ~/.cache/huggingface."
    say "Would never call sudo."
    echo
    say "The model is not downloaded here. Setup asks about that first,"
    say "after showing you the size."
    echo
    exit 0
fi

# ── Install ─────────────────────────────────────────────────────────────────
tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM

curl -fsSL --proto '=https' --tlsv1.2 -o "$tmp/$ASSET" "$URL" \
    || die "Couldn't download nightsmith from GitHub.
    Check your connection and try again — nothing was installed."

# Verify before installing, not after. A corrupted or wrong-architecture binary
# fails with "Killed: 9", which this audience cannot diagnose.
if curl -fsSL --proto '=https' -o "$tmp/SHA256SUMS" "$SUMS_URL" 2>/dev/null; then
    want="$(grep " $ASSET\$" "$tmp/SHA256SUMS" | awk '{print $1}')"
    got="$(shasum -a 256 "$tmp/$ASSET" | awk '{print $1}')"
    if [ -z "$want" ]; then
        warn "No checksum published for this asset; skipping verification."
    elif [ "$want" != "$got" ]; then
        die "The download doesn't match its published checksum.
    Expected $want
    Got      $got
    Nothing was installed. This is worth reporting."
    else
        ok "Checksum verified"
    fi
else
    warn "Couldn't fetch checksums; continuing without verification."
fi

chmod +x "$tmp/$ASSET"

# Apple Silicon refuses to execute unsigned native code. Catching a bad
# signature here produces a sentence; letting it through produces "Killed: 9".
#
# A valid signature is not enough: `codesign --verify` passes an ad-hoc or
# linker-only signature, which anyone can make. Require ours — Developer ID,
# this team — so a swapped binary with a valid-but-foreign signature is refused.
TEAM_ID="7229PY5Q9U"
if command -v codesign >/dev/null 2>&1; then
    codesign --verify --strict "$tmp/$ASSET" 2>/dev/null \
        || die "This download isn't correctly signed, and macOS will refuse to
    run it. Nothing was installed. This is worth reporting."
    sig="$(codesign -dvv "$tmp/$ASSET" 2>&1)"   # -dvv: -dv omits Authority
    case "$sig" in
        *"Authority=Developer ID Application:"*"TeamIdentifier=$TEAM_ID"*) ;;
        *) die "This download is signed, but not by nightsmith's developer.
    Nothing was installed. This is worth reporting." ;;
    esac
fi

mkdir -p "$BIN_DIR"
mv "$tmp/$ASSET" "$BIN"
size="$(du -h "$BIN" | awk '{print $1}')"
ok "Installed to $BIN   ($size, no password needed)"

if [ "$need_path_edit" = "1" ]; then
    case "$RC" in
        */config.fish)
            mkdir -p "$(dirname "$RC")"
            printf '\n# added by nightsmith\nfish_add_path %s\n' "$BIN_DIR" >> "$RC" ;;
        *)
            printf '\n# added by nightsmith\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >> "$RC" ;;
    esac
    ok "Added $BIN_DIR to your PATH in $(basename "$RC")"
    PATH="$BIN_DIR:$PATH"
    export PATH
fi

[ "$NO_RUN" = "1" ] && { echo; say "Run 'nightsmith' when you're ready."; exit 0; }

echo
say "Starting setup…"
echo

# Running the tool is what spares the user "now restart your terminal": a PATH
# edit does not reach the shell that is already open.
#
# But this script is usually running as `curl … | sh`, so stdin is the pipe,
# not the keyboard — and setup asks a question. Hand it the terminal, or it
# would read EOF and answer its own prompt.
if [ -r /dev/tty ]; then
    exec "$BIN" < /dev/tty
else
    # No terminal to hand over (CI, a non-interactive shell). Don't start
    # something interactive that cannot be answered.
    say "No terminal attached, so setup wasn't started."
    say "Run 'nightsmith' from a terminal when you're ready."
fi
