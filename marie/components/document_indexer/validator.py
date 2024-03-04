from abc import ABC, abstractmethod
from typing import Any


class EntityValidator(ABC):
    @abstractmethod
    def validate(self, value: Any) -> Any:
        """
        Validate the input value.

        :param value: The value to be validated.
        :return: The validated value if it is valid.
        :raises: ValueError if the value is invalid. The message should contain a description of the problem.
        """
        pass

    def __call__(self, value: str) -> Any:
        return self.validate(value)


class AddressValidator(EntityValidator):
    def validate(self, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError(f"Expected a string, but got {value}")

        import usaddress
        from i18naddress import InvalidAddressError, format_address, normalize_address

        modded = value.replace("\n", " ")

        mapping = {
            "Recipient": "name",
            "AddressNumber": "address1",
            "AddressNumberPrefix": "address1",
            "AddressNumberSuffix": "address1",
            "StreetName": "address1",
            "StreetNamePreDirectional": "address1",
            "StreetNamePreModifier": "address1",
            "StreetNamePreType": "address1",
            "StreetNamePostDirectional": "address1",
            "StreetNamePostModifier": "address1",
            "StreetNamePostType": "address1",
            # When corner addresses are given, you have two streets in an address
            "SecondStreetName": "address1",
            "SecondStreetNamePreDirectional": "address1",
            "SecondStreetNamePreModifier": "address1",
            "SecondStreetNamePreType": "address1",
            "SecondStreetNamePostDirectional": "address1",
            "SecondStreetNamePostModifier": "address1",
            "SecondStreetNamePostType": "address1",
            "CornerOf": "address1",
            "IntersectionSeparator": "address1",
            "LandmarkName": "address1",
            "USPSBoxGroupID": "address1",
            "USPSBoxGroupType": "address1",
            "USPSBoxID": "address1",
            "USPSBoxType": "address1",
            "BuildingName": "address2",
            "OccupancyType": "address2",
            "OccupancyIdentifier": "address2",
            "SubaddressIdentifier": "address2",
            "SubaddressType": "address2",
            "PlaceName": "city",
            "StateName": "state",
            "ZipCode": "zip_code",
            "ZipPlus4": "zip_code",
        }

        try:
            tagged_address, address_type = usaddress.tag(modded, tag_mapping=mapping)
        except (usaddress.RepeatedLabelError, UnicodeEncodeError):
            # See https://github.com/datamade/probableparsing/issues/2 for why we
            # catch the UnicodeEncodeError. Oy.
            raise ValueError(f"Unable to parse address (RepeatedLabelError): {value}")

        tagged_address.pop("NotAddress", None)

        if any([address_type == "Ambiguous", "CountryName" in tagged_address]):
            raise ValueError(
                f"Unable to parse address (Ambiguous address type): {value}"
            )

        name = tagged_address["name"] if "name" in tagged_address else ""
        address1 = tagged_address["address1"] if "address1" in tagged_address else ""
        address2 = tagged_address["address2"] if "address2" in tagged_address else ""
        city = tagged_address["city"] if "city" in tagged_address else ""
        state = tagged_address["state"] if "state" in tagged_address else ""
        zip_code = tagged_address["zip_code"] if "zip_code" in tagged_address else ""

        if "-" in zip_code and len(zip_code) < 10:
            zip_code = zip_code.split("-")[0]
        elif "-" in zip_code and len(zip_code) > 10:
            zip_code = zip_code[:5]
        elif len(zip_code) > 5:
            zip_code = zip_code[:5]

        try:
            address_dict = {
                # 'company_name': name,
                'country_code': "US",
                'country_area': state,
                'city': city,
                'postal_code': zip_code,
                'street_address': address1 + " " + address2,
                'state': state,
            }
            return format_address(normalize_address(address_dict))
        except InvalidAddressError as e:
            error_msg = ", ".join(e.errors)
            raise ValueError(f"Invalid address: {error_msg}") from e
