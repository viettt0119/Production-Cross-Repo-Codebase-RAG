from checkout.inventory import Inventory


def test_reserve_decrements():
    inv = Inventory({"a": 2})
    inv.reserve("a", 1)
    inv.reserve("a", 1)
