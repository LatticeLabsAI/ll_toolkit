#!/usr/bin/env python3
"""Test script to verify all cadling backends are working correctly."""

def test_imports():
    """Test that all backends can be imported."""
    print("Testing backend imports...")
    print("=" * 50)

    from cadling.backend.step.step_backend import STEPBackend
    from cadling.backend.stl.stl_backend import STLBackend
    from cadling.backend.brep.brep_backend import BRepBackend
    from cadling.backend.iges_backend import IGESBackend
    from cadling.backend.document_converter import DocumentConverter

    print("✓ STEP backend imported")
    print("✓ STL backend imported")
    print("✓ BRep backend imported")
    print("✓ IGES backend imported")
    print("✓ Document converter imported")
    print()

def test_capabilities():
    """Test backend capabilities."""
    print("Testing backend capabilities...")
    print("=" * 50)

    from cadling.backend.step.step_backend import STEPBackend
    from cadling.backend.stl.stl_backend import STLBackend
    from cadling.backend.brep.brep_backend import BRepBackend
    from cadling.backend.iges_backend import IGESBackend

    backends = [
        ("STEP", STEPBackend),
        ("STL", STLBackend),
        ("BRep", BRepBackend),
        ("IGES", IGESBackend),
    ]

    for name, backend_cls in backends:
        text_parsing = backend_cls.supports_text_parsing()
        rendering = backend_cls.supports_rendering()

        print(f"{name}:")
        print(f"  Text parsing: {'✓' if text_parsing else '✗'} {text_parsing}")
        print(f"  Rendering: {'✓' if rendering else '✗'} {rendering}")
        print()

def test_pythonocc():
    """Test pythonocc-core availability."""
    print("Testing pythonocc-core...")
    print("=" * 50)

    components = [
        ("STEP", "OCC.Core.STEPControl", "STEPControl_Reader"),
        ("IGES", "OCC.Core.IGESControl", "IGESControl_Reader"),
        ("BRep", "OCC.Core.BRepTools", "breptools"),
    ]

    for name, module_name, class_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {name}: pythonocc-core {module_name} available")
        except ImportError as e:
            print(f"✗ {name}: pythonocc-core not available - {e}")
    print()

def test_ll_stepnet():
    """Test ll_stepnet availability."""
    print("Testing ll_stepnet...")
    print("=" * 50)

    try:
        from ll_stepnet import STEPFeatureExtractor, STEPTokenizer, STEPTopologyBuilder
        print("✓ ll_stepnet is available")

        try:
            from ll_stepnet import __version__
            print(f"  Version: {__version__}")
        except:
            print("  Version: unknown")
    except ImportError as e:
        print(f"✗ ll_stepnet not available - {e}")
    print()

def test_ll_stepnet_integration():
    """Test STEP backend ll_stepnet integration."""
    print("Testing STEP backend ll_stepnet integration...")
    print("=" * 50)

    from cadling.backend.step.step_backend import STEPBackend
    from cadling.datamodel.backend_options import STEPBackendOptions
    from cadling.datamodel.base_models import CADInputDocument, InputFormat
    from io import BytesIO

    # Create minimal STEP content
    step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.step','2024-01-01',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
ENDSEC;
END-ISO-10303-21;
"""

    # Test with ll_stepnet enabled
    options_enabled = STEPBackendOptions(enable_ll_stepnet=True)
    in_doc = CADInputDocument(
        file="test.step",
        format=InputFormat.STEP,
        document_hash="test123"
    )
    stream = BytesIO(step_content.encode('utf-8'))

    backend = STEPBackend(in_doc, stream, options_enabled)

    if backend.use_ll_stepnet:
        print("✓ ll_stepnet integration is active")
        print(f"  ll_stepnet object: {backend.ll_stepnet}")
    else:
        print("✗ ll_stepnet integration not active (falling back to basic parsing)")

    # Test with ll_stepnet disabled
    options_disabled = STEPBackendOptions(enable_ll_stepnet=False)
    stream2 = BytesIO(step_content.encode('utf-8'))
    backend2 = STEPBackend(in_doc, stream2, options_disabled)

    if not backend2.use_ll_stepnet:
        print("✓ Basic parsing works when ll_stepnet disabled")
    else:
        print("⚠ ll_stepnet active even when disabled")

    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("CADLING BACKEND TEST SUITE")
    print("=" * 50 + "\n")

    try:
        test_imports()
        test_capabilities()
        test_pythonocc()
        test_ll_stepnet()
        test_ll_stepnet_integration()

        print("=" * 50)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
