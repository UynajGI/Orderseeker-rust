#!/usr/bin/env bash
# =============================================================
# setup_rust_projects.sh
# =============================================================
# Creates four independent Cargo projects from the flat .rs and
# _Cargo.toml source files in this directory, then builds them
# all in release mode.
#
# Usage:
#   chmod +x setup_rust_projects.sh
#   ./setup_rust_projects.sh
# =============================================================

set -e

MODELS=("xy_mc" "kuramoto" "vicsek" "rotor_sde")

for MODEL in "${MODELS[@]}"; do
    echo "──────────────────────────────────────────"
    echo "  Setting up: $MODEL"
    echo "──────────────────────────────────────────"

    # 1. Create Cargo project skeleton
    cargo new --bin "$MODEL" 2>/dev/null || true

    # 2. Copy source file → src/main.rs
    cp "${MODEL}.rs"          "${MODEL}/src/main.rs"

    # 3. Copy Cargo manifest (replaces the default one)
    cp "${MODEL}_Cargo.toml"  "${MODEL}/Cargo.toml"

    echo "  Project created: ./${MODEL}/"
done

echo ""
echo "══════════════════════════════════════════════"
echo "  Building all projects in release mode ..."
echo "══════════════════════════════════════════════"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "  cargo build --release  (${MODEL})"
    (cd "$MODEL" && cargo build --release)
    echo "  ✓ ${MODEL}/target/release/${MODEL}"
done

echo ""
echo "══════════════════════════════════════════════"
echo "  All builds succeeded!"
echo ""
echo "  Run each simulation:"
for MODEL in "${MODELS[@]}"; do
    echo "    cd ${MODEL} && ./target/release/${MODEL}"
done
echo "══════════════════════════════════════════════"
