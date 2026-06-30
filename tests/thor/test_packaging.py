import thor


def test_version():
    """Check to see that we can get the package version"""
    assert thor.__version__ is not None
