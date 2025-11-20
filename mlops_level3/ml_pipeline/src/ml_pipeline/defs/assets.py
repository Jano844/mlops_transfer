import dagster as dg


@dg.asset
def test_asset():
    print("Hello from test_asset!")
    return "This is a test asset"
