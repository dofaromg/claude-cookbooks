import tempfile
from pathlib import Path

import pytest

from mrliou_commercial_format import (
    CommercialNamingError,
    ComplianceError,
    ImmutableSignatureError,
    calculate_state_transition,
    get_layer_info,
    pack_commercial_package,
    parse_product_name,
    unpack_commercial_package,
    validate_metadata,
)


def test_parse_product_name_valid():
    """Tests parsing valid product names from strict naming architecture."""
    # Example 1: Mr.liou.TotalCore.Unity.v1.flpkg
    parsed1 = parse_product_name("Mr.liou.TotalCore.Unity.v1.flpkg")
    assert parsed1["vendor"] == "Mr.liou"
    assert parsed1["product_category"] == "TotalCore"
    assert parsed1["core_function"] == "Unity"
    assert parsed1["version"] == "v1"
    assert parsed1["extension"] == "flpkg"

    # Example 2: Mr.liou.LoRA.ResonanceBuilder.v1.pcode
    parsed2 = parse_product_name("Mr.liou.LoRA.ResonanceBuilder.v1.pcode")
    assert parsed2["vendor"] == "Mr.liou"
    assert parsed2["product_category"] == "LoRA"
    assert parsed2["core_function"] == "ResonanceBuilder"
    assert parsed2["version"] == "v1"
    assert parsed2["extension"] == "pcode"

    # Example 3: FlowPersona.CreationLoop.Core.v1.flpkg
    parsed3 = parse_product_name("FlowPersona.CreationLoop.Core.v1.flpkg")
    assert parsed3["vendor"] == "FlowPersona"
    assert parsed3["product_category"] == "CreationLoop"
    assert parsed3["core_function"] == "Core"
    assert parsed3["version"] == "v1"
    assert parsed3["extension"] == "flpkg"

    # Example 4: ZhiZhang.TotalCore.SystemPack.v1_with_FlowPassport_EdgeLink_Package_v1.zip
    parsed4 = parse_product_name(
        "ZhiZhang.TotalCore.SystemPack.v1_with_FlowPassport_EdgeLink_Package_v1.zip"
    )
    assert parsed4["vendor"] == "ZhiZhang"
    assert parsed4["product_category"] == "TotalCore"
    assert parsed4["core_function"] == "SystemPack"
    assert parsed4["version"] == "v1_with_FlowPassport_EdgeLink_Package_v1"
    assert parsed4["extension"] == "zip"


def test_parse_product_name_invalid():
    """Tests that invalid product names raise CommercialNamingError."""
    bad_names = [
        "Mr.liou.TotalCore.Unity.flpkg",  # Missing part
        "Mr.liou.TotalCore.Unity.v1.invalid_ext",  # Invalid extension
        "JustName.flpkg",  # Too short
        ".flpkg",  # No stem
        "Mr.liou.TotalCore.Unity.v1.sub.flpkg",  # Too many parts (not 5-part architecture)
    ]
    for name in bad_names:
        with pytest.raises(CommercialNamingError):
            parse_product_name(name)


def test_validate_metadata_success():
    """Tests metadata validation with completely valid signatures."""
    valid_meta = {
        "format": "flpkg/1.0",
        "origin_signature": "MrLiouWord",
        "philosophy": "怎麼過去，就怎麼回來",
        "created_at": "2026-08-08T10:31:15Z",
        "encryption_enabled": True,
        "layer_gravity": "L0-L7",
    }
    # Should complete without raising any exception
    validate_metadata(valid_meta)


def test_validate_metadata_violations():
    """Tests metadata validation raises ImmutableSignatureError on violations."""
    base_meta = {
        "format": "flpkg/1.0",
        "origin_signature": "MrLiouWord",
        "philosophy": "怎麼過去，就怎麼回來",
        "created_at": "2026-08-08T10:31:15Z",
        "encryption_enabled": True,
        "layer_gravity": "L0-L7",
    }

    # 1. Missing key
    meta_missing = base_meta.copy()
    del meta_missing["philosophy"]
    with pytest.raises(ImmutableSignatureError, match="Missing required metadata keys"):
        validate_metadata(meta_missing)

    # 2. Wrong origin_signature
    meta_sig = base_meta.copy()
    meta_sig["origin_signature"] = "NotMrLiou"
    with pytest.raises(ImmutableSignatureError, match="Metadata violation on key"):
        validate_metadata(meta_sig)

    # 3. Wrong philosophy
    meta_phil = base_meta.copy()
    meta_phil["philosophy"] = "Some other philosophy"
    with pytest.raises(ImmutableSignatureError, match="Metadata violation on key"):
        validate_metadata(meta_phil)

    # 4. Wrong layer gravity
    meta_grav = base_meta.copy()
    meta_grav["layer_gravity"] = "L1-L6"
    with pytest.raises(ImmutableSignatureError, match="Metadata violation on key"):
        validate_metadata(meta_grav)

    # 5. Invalid format pattern
    meta_fmt = base_meta.copy()
    meta_fmt["format"] = "flpkg_without_version"
    with pytest.raises(ImmutableSignatureError, match="Invalid format metadata format"):
        validate_metadata(meta_fmt)


def test_get_layer_info():
    """Tests retrieving technical definitions and components for L0-L7 layers."""
    # L0 ROOT
    l0 = get_layer_info(0)
    assert l0["name"] == "L0: ROOT"
    assert "MrLiouWord" in l0["components"][0]

    # L7 LOOP
    l7 = get_layer_info(7)
    assert l7["name"] == "L7: LOOP"
    assert "Liou 閉合定律" in l7["components"]

    # Invalid Layer
    with pytest.raises(ValueError, match="Invalid layer level"):
        get_layer_info(8)


def test_calculate_state_transition():
    """Tests State[n+1] = State[n] + dP_0 formulation."""
    state_n = 42.0
    dp_0 = 3.5
    state_n_plus_1 = calculate_state_transition(state_n, dp_0)
    assert state_n_plus_1 == 45.5

    # Reversibility
    assert state_n_plus_1 - dp_0 == state_n


def test_pack_and_unpack_roundtrip():
    """Tests packing files and unpacking them cleanly with validation and complete reversibility."""
    manifest = {
        "format": "flpkg/1.0",
        "origin_signature": "MrLiouWord",
        "philosophy": "怎麼過去，就怎麼回來",
        "created_at": "2026-08-08T10:31:15Z",
        "encryption_enabled": True,
        "layer_gravity": "L0-L7",
    }

    file_contents = {
        "seed.fltnz": "⋄fx.adj.112 ∴ ⋄fx.noun.024",
        "logic.pcode": "MOV FX.ADJ.112\nCALL FX.FLOW.007",
    }

    # Use a temporary directory for tests
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        package_file = tmp_path / "Mr.liou.TotalCore.Unity.v1.flpkg"
        extract_dir = tmp_path / "extracted"

        # Pack
        packed_path = pack_commercial_package(str(package_file), manifest, file_contents)
        assert Path(packed_path).exists()

        # Unpack & Validate
        unpacked_manifest, unpacked_files = unpack_commercial_package(packed_path, str(extract_dir))

        assert unpacked_manifest == manifest
        assert "manifest.json" in unpacked_files
        assert "seed.fltnz" in unpacked_files
        assert "logic.pcode" in unpacked_files

        # Check content is preserved exactly (no distortion/lossless)
        unpacked_seed = (extract_dir / "seed.fltnz").read_text(encoding="utf-8")
        assert unpacked_seed == file_contents["seed.fltnz"]

        unpacked_pcode = (extract_dir / "logic.pcode").read_text(encoding="utf-8")
        assert unpacked_pcode == file_contents["logic.pcode"]


def test_unpack_missing_manifest():
    """Tests unpacking a package with a missing manifest raises ImmutableSignatureError."""
    # Write an invalid ZIP (no manifest)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        package_file = tmp_path / "Mr.liou.TotalCore.Unity.v1.flpkg"
        extract_dir = tmp_path / "extracted"

        # Create zip with no manifest
        import zipfile

        with zipfile.ZipFile(package_file, "w") as zipf:
            zipf.writestr("somefile.txt", "hello")

        with pytest.raises(
            ImmutableSignatureError, match="Neither manifest.json nor meta.json found"
        ):
            unpack_commercial_package(str(package_file), str(extract_dir))


def test_pack_reserved_filenames_raises_error():
    """Tests that packing with manifest.json or meta.json in file_contents raises ComplianceError."""
    manifest = {
        "format": "flpkg/1.0",
        "origin_signature": "MrLiouWord",
        "philosophy": "怎麼過去，就怎麼回來",
        "created_at": "2026-08-08T10:31:15Z",
        "encryption_enabled": True,
        "layer_gravity": "L0-L7",
    }
    file_contents = {"manifest.json": "{}"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        package_file = tmp_path / "Mr.liou.TotalCore.Unity.v1.flpkg"

        with pytest.raises(ComplianceError, match="Conflicting reserved file"):
            pack_commercial_package(str(package_file), manifest, file_contents)
