"""Tests for the PropertySet class."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from laser.core import PropertySet


class TestPropertySet(unittest.TestCase):
    """Tests for the PropertySet class."""

    def test_empty_property_set(self):
        """Test an empty PropertySet."""
        # assert that the PropertySet is empty
        ps = PropertySet()
        assert len(ps) == 0

    def test_single_dict_property_set(self):
        """Test initialization from a single dictionary."""
        # assert that the PropertySet is initialized with a single dictionary
        ps = PropertySet({"a": 1, "b": 2})
        assert ps.a == 1
        assert ps.b == 2

    def test_single_property_set(self):
        """Test initialization from a single PropertySet."""
        # assert that the PropertySet is initialized with a single PropertySet
        ps = PropertySet(PropertySet({"a": 1, "b": 2}))
        assert ps.a == 1
        assert ps.b == 2

    def test_multiple_dict_property_set(self):
        """Test initialization from multiple dictionaries."""
        # assert that the PropertySet is initialized with multiple dictionaries
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3
        assert ps.d == 4

    def test_multiple_property_set(self):
        """Test initialization from multiple PropertySets."""
        # assert that the PropertySet is initialized with multiple PropertySets
        ps = PropertySet(PropertySet({"a": 1, "b": 2}), PropertySet({"c": 3, "d": 4}))
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3
        assert ps.d == 4

    def test_mixed_property_set(self):
        """Test initialization from a mix of dictionaries and PropertySets."""
        # assert that the PropertySet is initialized with a mix of dictionaries and PropertySets
        ps = PropertySet({"a": 1, "b": 2}, PropertySet({"c": 3, "d": 4}))
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3
        assert ps.d == 4

    # Test += PropertySet, happy path (no overlapping keys)
    def test_iadd_propertyset_happy_path(self):
        ps = PropertySet({"a": 1, "b": 2, "c": 3})
        ps += PropertySet({"d": 1, "e": 4, "f": 9})
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3
        assert ps.d == 1
        assert ps.e == 4
        assert ps.f == 9

    # Test += dictionary, happy path (no overlapping keys)
    def test_iadd_dictionary_happy_path(self):
        ps = PropertySet({"a": 1, "b": 2, "c": 3})
        ps += {"d": 1, "e": 4, "f": 9}
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3
        assert ps.d == 1
        assert ps.e == 4
        assert ps.f == 9

    # Test += PropertySet with overlapping keys throws ValueError
    def test_iadd_propertyset_overlapping(self):
        ps = PropertySet({"a": 1, "b": 2, "c": 3})
        with pytest.raises(ValueError, match="Cannot override existing value for 'c'."):
            ps += PropertySet({"c": 13, "d": 1, "e": 4, "f": 9})

    # Test += dictionary with overlapping keys throws ValueError
    def test_iadd_dictionary_overlapping(self):
        ps = PropertySet({"a": 1, "b": 2, "c": 3})
        with pytest.raises(ValueError, match="Cannot override existing value for 'c'."):
            ps += {"c": 13, "d": 1, "e": 4, "f": 9}

    # Test += PropertySet to an empty PropertySet
    def test_iadd_empty_propertyset(self):
        ps = PropertySet()
        ps += PropertySet({"d": 1, "e": 4, "f": 9})
        assert ps.d == 1
        assert ps.e == 4
        assert ps.f == 9

    # Test += dictionary to an empty PropertySet
    def test_iadd_empty_dictionary(self):
        ps = PropertySet()
        ps += {"d": 1, "e": 4, "f": 9}
        assert ps.d == 1
        assert ps.e == 4
        assert ps.f == 9

    # Test += empty PropertySet
    def test_iadd_propertyset_empty_propertyset(self):
        ps = PropertySet({"a": 1, "b": 2, "c": 3})
        ps += PropertySet()
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3

    # Test += empty dictionary
    def test_iadd_propertyset_empty_dictionary(self):
        ps = PropertySet({"a": 1, "b": 2, "c": 3})
        ps += {}
        assert ps.a == 1
        assert ps.b == 2
        assert ps.c == 3

    # Test PropertySet << PropertySet, happy path
    def test_lshift_propertyset_propertyset(self):
        ps1 = PropertySet({"a": 2, "b": 71, "c": 828})
        ps2 = ps1 << PropertySet({"b": 828, "c": 427})
        assert ps2.a == 2
        assert ps2.b == 828
        assert ps2.c == 427

    # Test PropertySet << dictionary, happy path
    def test_lshift_propertyset_dictionary(self):
        ps1 = PropertySet({"a": 2, "b": 71, "c": 828})
        ps2 = ps1 << {"b": 828, "c": 427}
        assert ps2.a == 2
        assert ps2.b == 828
        assert ps2.c == 427

    # Test PropertySet << empty PropertySet
    def test_lshift_propertyset_empty_propertyset(self):
        ps1 = PropertySet({"a": 2, "b": 71, "c": 828})
        ps2 = ps1 << PropertySet()
        assert ps2.a == 2
        assert ps2.b == 71
        assert ps2.c == 828

    # Test PropertySet << empty dictionary
    def test_lshift_propertyset_empty_dictionary(self):
        ps1 = PropertySet({"a": 2, "b": 71, "c": 828})
        ps2 = ps1 << {}
        assert ps2.a == 2
        assert ps2.b == 71
        assert ps2.c == 828

    # Test empty PropertySet << PropertySet raises ValueError
    def test_lshift_empty_propertyset_propertyset(self):
        with pytest.raises(ValueError, match="Cannot override missing key 'b'."):
            _ = PropertySet() << PropertySet({"b": 828, "c": 427})

    # Test empty PropertySet << dictionary raises ValueError
    def test_lshift_empty_propertyset_dictionary(self):
        with pytest.raises(ValueError, match="Cannot override missing key 'b'."):
            _ = PropertySet() << {"b": 828, "c": 427}

    # Test PropertySet << PropertySet with new keys raises ValueError
    def test_lshift_propertyset_propertyset_new_keys(self):
        with pytest.raises(ValueError, match="Cannot override missing key 'd'."):
            _ = PropertySet({"a": 2, "b": 71, "c": 828}) << PropertySet({"b": 828, "c": 427, "d": 12474619})

    # Test PropertySet << dictionary with new keys raises ValueError
    def test_lshift_propertyset_dictionary_new_keys(self):
        with pytest.raises(ValueError, match="Cannot override missing key 'd'."):
            _ = PropertySet({"a": 2, "b": 71, "c": 828}) << {"b": 828, "c": 427, "d": 12474619}

    # Test PropertySet << PropertySet, happy path
    def test_ilshift_propertyset_propertyset(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps <<= PropertySet({"b": 828, "c": 427})
        assert ps.a == 2
        assert ps.b == 828
        assert ps.c == 427

    # Test PropertySet << dictionary, happy path
    def test_ilshift_propertyset_dictionary(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps <<= {"b": 828, "c": 427}
        assert ps.a == 2
        assert ps.b == 828
        assert ps.c == 427

    # Test PropertySet << empty PropertySet
    def test_ilshift_propertyset_empty_propertyset(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps <<= PropertySet()
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test PropertySet << empty dictionary
    def test_ilshift_propertyset_empty_dictionary(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps <<= {}
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test empty PropertySet << PropertySet raises ValueError
    def test_ilshift_empty_propertyset_propertyset(self):
        ps = PropertySet()
        with pytest.raises(ValueError, match="Cannot override missing key 'b'."):
            ps <<= PropertySet({"b": 828, "c": 427})

    # Test empty PropertySet << dictionary raises ValueError
    def test_ilshift_empty_propertyset_dictionary(self):
        ps = PropertySet()
        with pytest.raises(ValueError, match="Cannot override missing key 'b'."):
            ps <<= {"b": 828, "c": 427}

    # Test PropertySet << PropertySet with new keys raises ValueError
    def test_ilshift_propertyset_propertyset_new_keys(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        with pytest.raises(ValueError, match="Cannot override missing key 'd'."):
            ps <<= PropertySet({"b": 828, "c": 427, "d": 12474619})

    # Test PropertySet << dictionary with new keys raises ValueError
    def test_ilshift_propertyset_dictionary_new_keys(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        with pytest.raises(ValueError, match="Cannot override missing key 'd'."):
            ps <<= {"b": 828, "c": 427, "d": 12474619}

    # Test PropertySet | PropertySet
    def test_or_propertyset_propertyset(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828}) | PropertySet({"b": 828, "c": 427, "d": 125})
        assert ps.a == 2
        assert ps.b == 828
        assert ps.c == 427
        assert ps.d == 125

    # Test PropertySet | dictionary
    def test_or_propertyset_dictionary(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828}) | {"b": 828, "c": 427, "d": 125}
        assert ps.a == 2
        assert ps.b == 828
        assert ps.c == 427
        assert ps.d == 125

    # Test PropertySet | empty PropertySet
    def test_or_propertyset_empty_propertyset(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828}) | PropertySet()
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test PropertySet | empty dictionary
    def test_or_propertyset_empty_dictionary(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828}) | {}
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test empty PropertySet | PropertySet
    def test_or_empty_propertyset_propertyset(self):
        ps = PropertySet() | PropertySet({"a": 2, "b": 71, "c": 828})
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test empty PropertySet | dictionary
    def test_or_empty_propertyset_dictionary(self):
        ps = PropertySet() | {"a": 2, "b": 71, "c": 828}
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test PropertySet | PropertySet
    def test_ior_propertyset_propertyset(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps |= PropertySet({"b": 828, "c": 427, "d": 125})
        assert ps.a == 2
        assert ps.b == 828
        assert ps.c == 427
        assert ps.d == 125

    # Test PropertySet | dictionary
    def test_ior_propertyset_dictionary(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps |= {"b": 828, "c": 427, "d": 125}
        assert ps.a == 2
        assert ps.b == 828
        assert ps.c == 427
        assert ps.d == 125

    # Test PropertySet | empty PropertySet
    def test_ior_propertyset_empty_propertyset(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps |= PropertySet()
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test PropertySet | empty dictionary
    def test_ior_propertyset_empty_dictionary(self):
        ps = PropertySet({"a": 2, "b": 71, "c": 828})
        ps |= {}
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test empty PropertySet | PropertySet
    def test_ior_empty_propertyset_propertyset(self):
        ps = PropertySet()
        ps |= PropertySet({"a": 2, "b": 71, "c": 828})
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    # Test empty PropertySet | dictionary
    def test_ior_empty_propertyset_dictionary(self):
        ps = PropertySet()
        ps |= {"a": 2, "b": 71, "c": 828}
        assert ps.a == 2
        assert ps.b == 71
        assert ps.c == 828

    def test_add_property_set_new(self):
        """Test that PropertySet + PropertySet creates a new PropertySet _and_ does not alter the existing PropertySets"""
        # assert that PropertySet + PropertySet creates a new PropertySet _and_ does not alter the existing PropertySets
        ps1 = PropertySet({"a": 1, "b": 2})
        ps2 = PropertySet({"b": 3, "c": 4})
        ps3 = ps1 + ps2
        assert ps1.a == 1
        assert ps1.b == 2
        assert ps2.b == 3
        assert ps2.c == 4
        assert ps3.a == 1
        assert ps3.b == 3
        assert ps3.c == 4

    def test_str(self):
        """Test the __str__ method of the PropertySet class."""
        # assert that the __str__ method returns the expected string
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert str(ps) == json.dumps({"a": 1, "b": 2, "c": 3, "d": 4}, indent=4)

    def test_repr(self):
        """Test the __repr__ method of the PropertySet class."""
        # assert that the __repr__ method returns the expected string
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert repr(ps) == f"PropertySet({ {'a': 1, 'b': 2, 'c': 3, 'd': 4}!s})"

    def test_contains(self):
        """Test the __contains__ method of the PropertySet class."""
        # assert that the __contains__ method returns the expected results
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert "a" in ps
        assert "b" in ps
        assert "c" in ps
        assert "d" in ps
        assert "e" not in ps

    def test_to_dict(self):
        """Test the to_dict method of the PropertySet class."""
        # assert that the to_dict method returns the expected dictionary
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert ps.to_dict() == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_to_dict_nested(self):
        """Test the to_dict method of the PropertySet class with nested PropertySets."""
        # assert that the to_dict method returns the expected dictionary with nested PropertySets
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4, "e": PropertySet({"f": 5, "g": 6})})
        assert ps.to_dict() == {"a": 1, "b": 2, "c": 3, "d": 4, "e": {"f": 5, "g": 6}}

    # Test save() method on temporary file
    def test_save(self):
        """Test the save method of the PropertySet class."""
        # assert that the save method writes the expected string to the file
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            ps.save(filename)
            assert filename.read_text() == str(ps)

    # Test load() method on temporary file
    def test_load(self):
        """Test the load method of the PropertySet class."""
        # assert that the load method reads the expected string from the file
        ps = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            ps.save(filename)
            assert PropertySet.load(filename) == ps

    def test_item_access(self):
        """Test item access, e.g., ``ps[key]``, in the PropertySet class."""
        ps = PropertySet({"a": 1, "b": 2.7182818285}, {"c": "three", "d": np.uint32(42)})
        assert ps["a"] == 1
        assert ps["b"] == 2.7182818285
        assert ps["c"] == "three"
        assert ps["d"] == np.uint32(42)
        ps.ps = PropertySet({"e": 2.7182818285})
        assert ps["ps"] == PropertySet({"e": 2.7182818285})
        assert ps.ps["e"] == 2.7182818285

    def test_item_set(self):
        """Test item set, e.g., ``ps[key] = value``, in the PropertySet class."""
        ps = PropertySet()
        ps["a"] = 1
        ps["b"] = 2.7182818285
        ps["c"] = "three"
        ps["d"] = np.uint32(42)
        assert ps.a == 1
        assert ps.b == 2.7182818285
        assert ps.c == "three"
        assert ps.d == np.uint32(42)
        ps["ps"] = PropertySet({"e": 2.7182818285})
        assert ps.ps == PropertySet({"e": 2.7182818285})
        assert ps.ps["e"] == 2.7182818285

    def test_single_list_property_set(self):
        """Test initialization from a single list."""
        # assert that the PropertySet is initialized with a single list
        ps = PropertySet([("a", 1), ("b", 2)])
        assert ps.a == 1
        assert ps.b == 2

    def test_single_tuple_property_set(self):
        """Test initialization from a single tuple."""
        # assert that the PropertySet is initialized with a single tuple
        ps = PropertySet((("a", 1), ("b", 2)))
        assert ps.a == 1
        assert ps.b == 2

    def test_init_with_unsupported_type(self):
        """Test initialization with an unsupported type (e.g., string) raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type 'str' for PropertySet initialization"):
            PropertySet("not a valid bag")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
