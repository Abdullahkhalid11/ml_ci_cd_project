def test_basic():
    assert True

def test_import():
    from src.app import app
    assert app is not None