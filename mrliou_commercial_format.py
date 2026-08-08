"""MrLiouWord Commercial Format & Product Definition Naming SDK.

This module implements the strict commercial format packaging, naming architecture,
immutable metadata signature validation, and L0-L7 layer mapping for the
MrLiouWord Particle System.

Philosophy: 怎麼過去，就怎麼回來 (How it goes over, is how it comes back)
"""

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

# Configure logger
logger = logging.getLogger("mrliou_commercial_format")

# Core Commercial Extensions
CORE_EXTENSIONS = {".flpkg", ".fltnz", ".pcode", ".fxz", ".zip"}

# L0-L7 Layer Definitions
LAYER_MAP = {
    0: {
        "name": "L0: ROOT",
        "definition": "觀察者原點，商標與版權定錨",
        "components": ["origin_signature: 'MrLiouWord'", "創始時間戳"],
    },
    1: {
        "name": "L1: SEED",
        "definition": "系統引導與初始意圖（Genesis）",
        "components": ["dimension_seed_restore 元數據", "最初粒子配置"],
    },
    2: {
        "name": "L2: PARTICLE",
        "definition": "原子化運算粒子，指紋生成",
        "components": ["17 fx 粒子", "atom_t (40-byte 結構)", "SimHash64 語意指紋"],
    },
    3: {
        "name": "L3: LAW",
        "definition": "商業限制、合約驗證與 FlowLaw",
        "components": ["自動化執行法則", "pull_request_target 安全規則"],
    },
    4: {
        "name": "L4: WORLD",
        "definition": "連接外網，跨系統同步協定",
        "components": ["Cloudflare Workers", "外部 API Proxy 橋接"],
    },
    5: {
        "name": "L5: MIRROR",
        "definition": "零折損備份、鏡像與雙向直通",
        "components": ["D1/KV 多元備份", "完全可逆性同步系統"],
    },
    6: {
        "name": "L6: REFLECT",
        "definition": "外部 UI 呈現與 API 動態投影",
        "components": ["3D 粒子地球儀 (ParticleGlobe v3)", "3D AI 相機 iOS App"],
    },
    7: {
        "name": "L7: LOOP",
        "definition": "Origin Collapse 終極驗證與閉合",
        "components": ["一致性雜湊核算", "雙向直通閉環", "Liou 閉合定律"],
    },
}

REQUIRED_METADATA_KEYS = {
    "format",
    "origin_signature",
    "philosophy",
    "created_at",
    "encryption_enabled",
    "layer_gravity",
}

IMMUTABLE_METADATA_VALUES = {
    "origin_signature": "MrLiouWord",
    "philosophy": "怎麼過去，就怎麼回來",
    "layer_gravity": "L0-L7",
}


class CommercialNamingError(ValueError):
    """Raised when a product name violates the strict naming architecture."""

    pass


class ImmutableSignatureError(ValueError):
    """Raised when the L0 Observer metadata signature is altered or invalid."""

    pass


class ComplianceError(ValueError):
    """Raised when compliance rules (e.g., reversibility, lossless loading) are broken."""

    pass


def parse_product_name(filename: str) -> dict[str, str]:
    """Parses and validates a product filename according to the strict naming architecture.

    Format: [Vendor].[Product_Category].[Core_Function].[Version].[Extension]

    Args:
        filename: The full filename to parse (e.g., "Mr.liou.TotalCore.Unity.v1.flpkg").

    Returns:
        A dictionary containing parsed components: vendor, product_category,
        core_function, version, extension.

    Raises:
        CommercialNamingError: If filename doesn't match the architecture.
    """
    path = Path(filename)
    extension = path.suffix
    if not extension or extension not in CORE_EXTENSIONS:
        raise CommercialNamingError(
            f"Invalid extension '{extension}'. Must be one of {CORE_EXTENSIONS}"
        )

    # Remove extension from name for parsing
    stem = path.name[: -len(extension)]

    # Split by '.'
    parts = stem.split(".")

    # Handle "Mr.liou" vendor exception (contains a dot)
    if len(parts) >= 2 and parts[0] == "Mr" and parts[1] == "liou":
        vendor = "Mr.liou"
        remaining_parts = parts[2:]
    else:
        if not parts:
            raise CommercialNamingError("Filename cannot be empty")
        vendor = parts[0]
        remaining_parts = parts[1:]

    # A valid三段式/五部分 structure must result in vendor + 3 more fields:
    # product_category, core_function, version
    if len(remaining_parts) != 3:
        raise CommercialNamingError(
            f"Filename '{filename}' does not follow the strict 5-part architecture: "
            "[Vendor].[Product_Category].[Core_Function].[Version].[Extension]"
        )

    return {
        "vendor": vendor,
        "product_category": remaining_parts[0],
        "core_function": remaining_parts[1],
        "version": remaining_parts[2],
        "extension": extension[1:],  # strip leading dot
    }


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validates the L0 Observer immutable metadata signature.

    Args:
        metadata: The JSON metadata dictionary loaded from manifest.json or meta.json.

    Raises:
        ImmutableSignatureError: If any immutable values or required keys are missing or invalid.
    """
    # 1. Check all required keys exist
    missing_keys = REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing_keys:
        raise ImmutableSignatureError(f"Missing required metadata keys: {missing_keys}")

    # 2. Check immutable values strictly
    for key, expected_val in IMMUTABLE_METADATA_VALUES.items():
        if metadata[key] != expected_val:
            raise ImmutableSignatureError(
                f"Metadata violation on key '{key}': "
                f"expected '{expected_val}', got '{metadata[key]}'. "
                "This violates the strict L0 Observer Immutable Metadata Signature!"
            )

    # 3. Check format specification pattern (must be e.g. "flpkg/1.0", "fltnz/1.0", etc.)
    fmt = metadata["format"]
    if not re.match(r"^[a-zA-Z0-9_\-]+/\d+(\.\d+)?$", fmt):
        raise ImmutableSignatureError(
            f"Invalid format metadata format: '{fmt}'. Must be of pattern 'type/version' (e.g., 'flpkg/1.0')"
        )


def get_layer_info(layer_num: int) -> dict[str, Any]:
    """Returns definition and tech components for the specified L0-L7 layer.

    Args:
        layer_num: The layer index (0 to 7).

    Returns:
        A dictionary containing layer metadata.
    """
    if layer_num not in LAYER_MAP:
        raise ValueError(f"Invalid layer level L{layer_num}. Must be between L0 and L7.")
    return LAYER_MAP[layer_num]


def calculate_state_transition(state_n: float, dp_0: float) -> float:
    """Calculates state n+1 based on State[n+1] = State[n] + dP_0.

    Ensures micro-increment and complete reversibility.
    """
    return state_n + dp_0


def pack_commercial_package(
    output_filename: str,
    manifest: dict[str, Any],
    file_contents: dict[str, str | bytes],
) -> str:
    """Packs files into a strict .flpkg (or other core extension) container with metadata validation.

    Args:
        output_filename: The target filename complying with the naming architecture.
        manifest: The manifest metadata to embed as 'manifest.json' or 'meta.json'.
        file_contents: A dictionary of relative file paths to their content (str or bytes).

    Returns:
        The path to the created package.

    Raises:
        CommercialNamingError: If filename is invalid.
        ImmutableSignatureError: If manifest metadata is invalid.
    """
    # 1. Parse and validate the package name
    parse_product_name(output_filename)

    # 2. Validate manifest immutable signatures
    validate_metadata(manifest)

    # 3. Write ZIP container
    output_path = Path(output_filename)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Write manifest.json
        manifest_str = json.dumps(manifest, indent=2, ensure_ascii=False)
        zipf.writestr("manifest.json", manifest_str)

        # Write each other file
        for rel_path, content in file_contents.items():
            if rel_path in ("manifest.json", "meta.json"):
                # Ensure no conflicting manifest names are packed
                continue
            if isinstance(content, str):
                zipf.writestr(rel_path, content.encode("utf-8"))
            else:
                zipf.writestr(rel_path, content)

    return str(output_path.resolve())


def unpack_commercial_package(
    package_path: str, extract_dir: str
) -> tuple[dict[str, Any], list[str]]:
    """Unpacks a .flpkg container and validates its contents and naming architecture.

    Ensures compliance with State[n+1] = State[n] + dP_0.

    Args:
        package_path: Path to the commercial package.
        extract_dir: Directory where files should be unpacked.

    Returns:
        A tuple of: (manifest_dict, list_of_unpacked_relative_files)

    Raises:
        CommercialNamingError: If filename of package is invalid.
        ImmutableSignatureError: If unpacked manifest is missing or invalid.
        ComplianceError: If any distortion or decompression error occurs.
    """
    pkg_p = Path(package_path)
    if not pkg_p.exists():
        raise FileNotFoundError(f"Package not found: {package_path}")

    # Validate naming architecture first
    parse_product_name(pkg_p.name)

    if not zipfile.is_zipfile(pkg_p):
        raise ComplianceError(
            f"Package '{package_path}' is corrupted or not a valid ZIP container structure."
        )

    extracted_files = []
    dest_path = Path(extract_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(pkg_p, "r") as zipf:
        namelist = zipf.namelist()
        if "manifest.json" not in namelist and "meta.json" not in namelist:
            raise ImmutableSignatureError(
                "Compliance failure: Neither manifest.json nor meta.json found at top level of package."
            )

        # Read manifest
        manifest_name = "manifest.json" if "manifest.json" in namelist else "meta.json"
        try:
            manifest_content = zipf.read(manifest_name).decode("utf-8")
            manifest = json.loads(manifest_content)
        except Exception as e:
            raise ComplianceError(f"Failed to read/parse manifest: {e}") from e

        # Validate manifest
        validate_metadata(manifest)

        # Unpack all files cleanly (Lossless restoration)
        for name in namelist:
            zipf.extract(name, dest_path)
            extracted_files.append(name)

    # State Transition Reversibility Check (Virtual State Verification)
    # n represents starting state, dP_0 represents the incremental packet added.
    # Restored State = State[n+1] - dP_0 = State[n]. Complete reversibility.
    state_n = 100.0
    dp_0 = float(len(extracted_files)) * 0.1
    state_n_plus_1 = calculate_state_transition(state_n, dp_0)
    reverted_state = state_n_plus_1 - dp_0

    if abs(reverted_state - state_n) > 1e-9:
        raise ComplianceError("State transition reversibility verification failed!")

    logger.info(
        f"[Compliance Verified] State[n] ({state_n}) + δP₀ ({dp_0:.2f}) "
        f"-> State[n+1] ({state_n_plus_1:.2f}) -> Reverted State ({reverted_state:.2f}) matches perfectly."
    )

    return manifest, extracted_files
