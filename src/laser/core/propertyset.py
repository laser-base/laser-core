"""Implements a PropertySet class that can be used to store properties in a dictionary-like object."""

import json
from pathlib import Path
from typing import Union


class PropertySet:
    """Dictionary-like parameter bag with attribute access and composable merge operators.

    `PropertySet` is the canonical way to bundle model parameters in LASER. Properties
    can be read and written both as attributes (`ps.beta`) and as items (`ps["beta"]`),
    and multiple `PropertySet`s compose via three operators with different
    add-vs-override semantics:

    | Operator | Behavior | Raises on key conflict |
    | --- | --- | --- |
    | `+= other` | Add new keys only | `ValueError` if `other` contains a key already in `self` |
    | `<<= other` | Override existing keys only | `ValueError` if `other` contains a key not in `self` |
    | `\\|= other` | Add or override (union-like) | Never |

    The `+`, `<<`, `\\|` non-mutating variants return a new `PropertySet` rather than
    modifying the left-hand side.

    Args (constructor):
        *bags (PropertySet | list | tuple | dict): One or more sources of `(key, value)`
            pairs used to seed the set. Keys must be strings; values can be any type.
            Multiple bags are merged left-to-right with later values winning on conflict.

    Raises:
        TypeError: If a `bag` is not one of the supported types.
        ValueError: If a key conflict is detected by `+=` or `<<=`.

    **Example** — basic construction, attribute and item access:

        from laser.core import PropertySet

        ps = PropertySet({"mything": 0.4, "that_other_thing": 42})
        ps["status"] = "susceptible"
        print(ps.mything)         # 0.4
        print(ps["status"])       # 'susceptible'
        print(len(ps))            # 3
        print(ps.to_dict())       # {'mything': 0.4, 'that_other_thing': 42, 'status': 'susceptible'}

    **Example** — composing parameter sets:

        base = PropertySet({"immunity": "high", "region": "north"})
        extras = PropertySet({"infectivity": 0.7})
        combined = base + extras            # new PropertySet, base unchanged
        base |= {"new_timer": 10}           # in-place union (add-or-override)
        base <<= {"region": "south"}        # in-place override; key must already exist

    **Example** — save and load:

        ps.save("properties.json")
        ps2 = PropertySet.load("properties.json")
    """

    def __init__(self, *bags: Union["PropertySet", list, tuple, dict]):
        """
        Initialize a PropertySet to manage properties in a dictionary-like structure.

        Parameters:
            bags: A sequence of key-value pairs (e.g., lists, tuples, dictionaries) to initialize the PropertySet. Keys must be strings, and values can be any type.
        """

        iterator_mapping = {
            type(self): lambda o: o.__dict__.items(),
            list: lambda o: o,
            tuple: lambda o: o,
            dict: lambda o: o.items(),
        }

        for bag in bags:
            if type(bag) not in iterator_mapping:
                raise TypeError(
                    f"Unsupported type '{type(bag).__name__}' for PropertySet initialization. Expected one of PropertySet, list[tuple], tuple[tuple], or dictionary."
                )
            it = iterator_mapping[type(bag)](bag)
            for key, value in it:
                setattr(self, key, value)

    def to_dict(self):
        """Convert the PropertySet to a dictionary."""
        result = {}

        for key, value in self.__dict__.items():
            if isinstance(value, PropertySet):
                result[key] = value.to_dict()
            else:
                result[key] = value

        return result

    def save(self, filename):
        """
        Save the PropertySet to a specified file.

        Parameters:
            filename (str): The path to the file where the PropertySet will be saved.
        """
        file = Path(filename)
        with file.open("w") as file:
            file.write(str(self))

        return

    def __getitem__(self, key):
        """
        Retrieve the attribute of the object with the given key (e.g., ``ps[key]``).

        Parameters:
            key (str): The name of the attribute to retrieve.

        Returns:
            Any (any): The value of the attribute with the specified key.

        Raises:
            AttributeError: If the attribute with the specified key does not exist.
        """

        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        Set the value of an attribute.
        This method allows setting an attribute of the instance using the
        dictionary-like syntax (e.g., ``ps[key] = value``).

        Parameters:
            key (str): The name of the attribute to set.
            value (any): The value to set for the attribute.
        """

        setattr(self, key, value)

    def __add__(self, other):
        """
        Add another PropertySet to this PropertySet.

        This method allows the use of the ``+`` operator to combine two PropertySet instances.

        Parameters:
            other (PropertySet): The other PropertySet instance to add.

        Returns:
            PropertySet (PropertySet): A new PropertySet instance that combines the properties of both instances.
        """

        return PropertySet(self, other)

    def __iadd__(self, other):
        """
        Implements the in-place addition (``+=``) operator for the class.

        This method allows the instance to be updated with attributes from another
        instance of the same class or from a dictionary. If `other` is an instance
        of the same class, its attributes are copied to the current instance. If
        `other` is a dictionary, its key-value pairs are added as attributes to
        the current instance.

        Parameters:
            other (Union[type(self), dict]): The object or dictionary to add to the current instance.

        Returns:
            self (PropertySet): The updated instance with the new attributes.

        Raises:
            TypeError: If `other` is neither an instance of the same class nor a dictionary.
            ValueError: If `other` contains keys already present in the PropertySet.
        """

        if not isinstance(other, type(self) | dict):
            raise TypeError(f"other must be a {type(self).__name__} or dict (got {type(other).__name__})")

        for key, value in (other.__dict__ if isinstance(other, type(self)) else other).items():
            if hasattr(self, key):
                raise ValueError(f"Cannot override existing value for '{key}'.")
            setattr(self, key, value)
        return self

    def __lshift__(self, other):
        """
        Implements the ``<<`` operator on PropertySet to override existing values with new values.

        Parameters:
            other (Union[type(self), dict]): The object or dictionary with overriding values.

        Returns:
            PropertySet (PropertySet): A new PropertySet with all the values of the first PropertySet with overrides from the second PropertySet.

        Raises:
            TypeError: If `other` is neither an instance of the same class nor a dictionary.
            ValueError: If `other` contains keys not present in the PropertySet.
        """

        result = PropertySet(self)
        result <<= other

        return result

    def __ilshift__(self, other):
        """
        Implements the ``<<=`` operator on PropertySet to override existing values with new values.

        Parameters:
            other (Union[type(self), dict]): The object or dictionary with overriding values.

        Returns:
            self (PropertySet): The updated instance with the overrides from other.

        Raises:
            TypeError: If `other` is neither an instance of the same class nor a dictionary.
            ValueError: If `other` contains keys not present in the PropertySet.
        """

        if not isinstance(other, type(self) | dict):
            raise TypeError(f"other must be a {type(self).__name__} or dict (got {type(other).__name__})")

        for key, value in (other.__dict__ if isinstance(other, type(self)) else other).items():
            if not hasattr(self, key):
                raise ValueError(f"Cannot override missing key '{key}'.")
            setattr(self, key, value)
        return self

    def __or__(self, other):
        """
        Implements the ``|`` operator on PropertySet to add new or override existing values with new values.

        Parameters:
            other (Union[type(self), dict]): The object or dictionary with overriding values.

        Returns:
            PropertySet (PropertySet): A new PropertySet with all the values of the first PropertySet with new or overriding values from the second PropertySet.

        Raises:
            TypeError: If `other` is neither an instance of the same class nor a dictionary.
        """

        result = PropertySet(self)
        result |= other

        return result

    def __ior__(self, other):
        """
        Implements the ``|=`` operator on PropertySet to override existing values with new values.

        Parameters:
            other (Union[type(self), dict]): The object or dictionary with overriding values.

        Returns:
            self (PropertySet): The updated instance with all the values of self with new or overriding values from other.

        Raises:
            TypeError: If `other` is neither an instance of the same class nor a dictionary.
        """

        if not isinstance(other, type(self) | dict):
            raise TypeError(f"other must be a {type(self).__name__} or dict (got {type(other).__name__})")

        for key, value in (other.__dict__ if isinstance(other, type(self)) else other).items():
            # no check on existence in self, all keys added or updated
            setattr(self, key, value)
        return self

    def __len__(self):
        """
        Return the number of attributes in the instance.

        This method returns the number of attributes stored in the instance's
        __dict__ attribute, which represents the instance's namespace.

        Returns:
            int (int): The number of attributes in the instance.
        """

        return len(self.__dict__)

    def __str__(self) -> str:
        """
        Returns a string representation of the object's dictionary.

        This method is used to provide a human-readable string representation
        of the object, which includes all the attributes stored in the object's
        `__dict__`.

        Returns:
            str: A string representation of the object's dictionary.
        """

        return json.dumps(self.to_dict(), indent=4)

    def __repr__(self) -> str:
        """
        Return a string representation of the PropertySet instance.

        The string representation includes the class name and the dictionary of
        the instance's attributes.

        Returns:
            str: A string representation of the PropertySet instance.
        """

        return f"PropertySet({self.to_dict()!s})"

    def __contains__(self, key):
        """
        Check if a key is in the property set.

        Parameters:
            key (str): The key to check for existence in the property set.

        Returns:
            bool (bool): True if the key exists in the property set, False otherwise.
        """

        return key in self.__dict__

    def __eq__(self, other):
        """
        Check if two PropertySet instances are equal.

        Parameters:
            other (PropertySet): The other PropertySet instance to compare.

        Returns:
            bool (bool): True if the two instances are equal, False otherwise.
        """

        return self.to_dict() == other.to_dict()

    @staticmethod
    def load(filename):
        """
        Load a PropertySet from a specified file.

        Parameters:
            filename (str): The path to the file where the PropertySet is saved.

        Returns:
            PropertySet (PropertySet): The PropertySet instance loaded from the file.
        """
        with Path(filename).open("r") as file:
            data = json.load(file)

        return PropertySet(data)
