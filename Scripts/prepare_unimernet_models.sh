#!/bin/bash

set -euo pipefail

readonly ENCODER_NAME="UniMERNetTinyEncoder-FP16"
readonly DECODER_NAME="UniMERNetTinyDecoder-CachedStep-SelfKV-FP16"
readonly ARTIFACTS_DIR="${SRCROOT}/Tools/model-conversion/artifacts"
readonly ENCODER_PACKAGE="${ARTIFACTS_DIR}/${ENCODER_NAME}.mlpackage"
readonly DECODER_PACKAGE="${ARTIFACTS_DIR}/${DECODER_NAME}.mlpackage"
readonly TOKENIZER_SOURCE="${SRCROOT}/EqnSnap/Resources/Models/UniMERNet/UniMERNetTokenizer.json"
readonly RESOURCES_DIR="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"

fail() {
    echo "error: $*" >&2
    exit 1
}

verify_file() {
    local package="$1"
    local relative_path="$2"
    local expected_hash="$3"
    local file="${package}/${relative_path}"

    [[ -f "${file}" ]] || fail "UniMERNet model file is missing: ${file}"

    local actual_hash
    actual_hash="$(shasum -a 256 "${file}" | awk '{print $1}')"
    [[ "${actual_hash}" == "${expected_hash}" ]] || fail \
        "UniMERNet model checksum mismatch: ${file} (expected ${expected_hash}, got ${actual_hash})"
}

encoder_exists=0
decoder_exists=0
[[ -d "${ENCODER_PACKAGE}" ]] && encoder_exists=1
[[ -d "${DECODER_PACKAGE}" ]] && decoder_exists=1

mkdir -p "${RESOURCES_DIR}"
rm -rf \
    "${RESOURCES_DIR}/${ENCODER_NAME}.mlmodelc" \
    "${RESOURCES_DIR}/${DECODER_NAME}.mlmodelc"
rm -f "${RESOURCES_DIR}/UniMERNetTokenizer.json"

if [[ ${encoder_exists} -eq 0 && ${decoder_exists} -eq 0 ]]; then
    echo "note: UniMERNet FP16 artifacts are absent; building pix2tex-only app."
    exit 0
fi

if [[ ${encoder_exists} -ne ${decoder_exists} ]]; then
    fail "UniMERNet artifacts are incomplete; both ${ENCODER_NAME}.mlpackage and ${DECODER_NAME}.mlpackage are required."
fi

[[ -f "${TOKENIZER_SOURCE}" ]] || fail "UniMERNet tokenizer is missing: ${TOKENIZER_SOURCE}"

verify_file "${ENCODER_PACKAGE}" "Manifest.json" \
    "173d28beaae9de4260f77da1ad4392c9cf046a47b17501ef4ae6f91b2719b645"
verify_file "${ENCODER_PACKAGE}" "Data/com.apple.CoreML/weights/weight.bin" \
    "4b1f28881bb1201cb7ddd039dfdc8cc7d91543c30f7c7ba0bb7de66d019176ce"
verify_file "${ENCODER_PACKAGE}" "Data/com.apple.CoreML/model.mlmodel" \
    "a5fe8b9c073c3fd6f1af11086ddaf278e165faa9a7053ca807d6a2adbb0867b4"
verify_file "${DECODER_PACKAGE}" "Manifest.json" \
    "e986c29d38bf3629937d63c26c203d567208438e1bc60e00629d16e12049180c"
verify_file "${DECODER_PACKAGE}" "Data/com.apple.CoreML/weights/weight.bin" \
    "af109b8aac77667e01caf302727f3c1b46fe90c994645dcedd3e8fb8ae000f54"
verify_file "${DECODER_PACKAGE}" "Data/com.apple.CoreML/model.mlmodel" \
    "e2697466c778366b3ca3ae1241af6dd0807bdb5e4c0d56898379e500ea87f338"

temporary_directory="$(mktemp -d "${TEMP_DIR:-/tmp}/eqnsnap-unimernet.XXXXXX")"
trap 'rm -rf "${temporary_directory}"' EXIT

compile_model() {
    local package="$1"
    local name="$2"
    local output_directory="${temporary_directory}/${name}"

    mkdir -p "${output_directory}"
    xcrun coremlcompiler compile "${package}" "${output_directory}"
    [[ -d "${output_directory}/${name}.mlmodelc" ]] || fail \
        "coremlcompiler did not produce ${name}.mlmodelc"
    ditto \
        "${output_directory}/${name}.mlmodelc" \
        "${RESOURCES_DIR}/${name}.mlmodelc"
}

compile_model "${ENCODER_PACKAGE}" "${ENCODER_NAME}"
compile_model "${DECODER_PACKAGE}" "${DECODER_NAME}"
install -m 0644 "${TOKENIZER_SOURCE}" "${RESOURCES_DIR}/UniMERNetTokenizer.json"

echo "note: Injected verified UniMERNet FP16 models into ${RESOURCES_DIR}"
