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

# zsh: .zshenv, not .zshrc. Every zsh reads .zshenv; only interactive ones
# read .zshrc, so a PATH line there is invisible to `zsh -c`, cron, editors and
# coding agents, which then answer "nightsmith not found" on a working install.
case "$SHELL" in
    */zsh)  RC="$HOME/.zshenv" ;;
    */bash) RC="$HOME/.bash_profile" ;;
    */fish) RC="$HOME/.config/fish/config.fish" ;;
    *)      RC="$HOME/.profile" ;;
esac

path_has_bin_dir() {
    case ":${PATH}:" in *":${BIN_DIR}:"*) return 0 ;; *) return 1 ;; esac
}
rc_has_bin_dir() {
    rel="${BIN_DIR#"$HOME"/}"
    [ -f "$RC" ] && grep -qF -e "$BIN_DIR" -e "\$HOME/$rel" -e "~/$rel" "$RC" 2>/dev/null
}

# Two separate questions:
#   need_path_edit   does the profile file need our line?
#   shell_lacks_bin  does the shell running this lack the directory right now?
# For zsh the first is asked of .zshenv alone. A PATH that already has
# ~/.local/bin usually got it from .zshrc — an earlier install, or pipx or uv —
# and .zshrc reaches interactive shells only, so the directory being on PATH
# here says nothing about `zsh -c` or a coding agent.
need_path_edit=0
shell_lacks_bin=0
path_has_bin_dir || shell_lacks_bin=1
case "$RC" in
    */.zshenv) rc_has_bin_dir || need_path_edit=1 ;;
    *) [ "$shell_lacks_bin" = "1" ] && ! rc_has_bin_dir && need_path_edit=1 ;;
esac

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
TEAM_ID="G3J29WY48M"
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
fi
if [ "$shell_lacks_bin" = "1" ]; then
    PATH="$BIN_DIR:$PATH"
    export PATH
fi

if [ "$NO_RUN" = "1" ]; then
    echo
    if [ "$shell_lacks_bin" = "1" ]; then
        say "This terminal doesn't know the command yet. Run this once,"
        say "or open a new terminal window:"
        echo
        say "    source $RC"
        echo
    fi
    say "Then run 'nightsmith' when you're ready."
    exit 0
fi

echo
say "Starting setup…"
echo

# Running the tool is what spares the user "now restart your terminal": a PATH
# edit does not reach the shell that is already open.
#
# But this script is usually running as `curl … | sh`, so stdin is the pipe,
# not the keyboard — and setup asks a question. Hand it the terminal, or it
# would read EOF and answer its own prompt.
# Setup's last screen lists `nightsmith start` and friends. A PATH line added
# just now reaches future shells only — no process can change its parent's
# environment — so the shell the user returns to would answer "command not
# found" one screen after being told it is ready.
#
# So: when the PATH had to be edited, hand the user a *new* login shell once
# setup is done. It reads the edited profile, the command works in it, and
# nobody has to be told to run anything. When there is no terminal to hand
# over, setup prints its commands by full path instead, which need no PATH at
# all. Either way the user never types a fix-up line.
#
# NIGHTSMITH_PATH_STATE tells the tool which of those two endings applies:
#   fresh-shell  a login shell follows, so plain `nightsmith …` will work
#   full-path    nothing follows; print ~/.local/bin/nightsmith …
# Unset means the PATH already had the directory and neither applies.
# `[ -r /dev/tty ]` is not enough: the file exists and looks readable even when
# the process has no controlling terminal, and opening it then fails with
# "Device not configured". Try the open itself.
if ! { : < /dev/tty; } 2>/dev/null; then
    # No terminal to hand over (CI, a non-interactive shell). Don't start
    # something interactive that cannot be answered.
    say "No terminal attached, so setup wasn't started."
    if [ "$shell_lacks_bin" = "1" ]; then
        # The short name will not work in a shell started before the PATH
        # edit, so give the path that works anywhere.
        say "Run '$(printf '%s' "$BIN" | sed "s|^$HOME|~|")' from a terminal when you're ready."
    else
        say "Run 'nightsmith' from a terminal when you're ready."
    fi
    exit 0
fi

if [ "$shell_lacks_bin" = "1" ]; then
    NIGHTSMITH_PATH_STATE=fresh-shell
else
    NIGHTSMITH_PATH_STATE=""
fi
export NIGHTSMITH_PATH_STATE

if ! "$BIN" < /dev/tty; then
    # Setup failed or was declined. Dropping someone into a nested shell after
    # that would be one surprise on top of another.
    exit 1
fi

if [ "$shell_lacks_bin" = "1" ]; then
    echo
    say "Opened a fresh shell here, so the nightsmith command works."
    say "(Type 'exit' to return to where you were.)"
    echo
    exec "$SHELL" -l < /dev/tty
fi
