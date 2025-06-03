from marshmallow import Schema, fields
from marshmallow_dataclass import NewType

from skypro.commands.simulator.cartesian import Point, Curve


"""
This handles the creation of Curve and Point types from YAML configuration (deserialization) 
"""


class PointSchema(Schema):
    x = fields.Float
    y = fields.Float
    # Somehow these aren't being validated as floats - Infinity is coming through as a string


class PointField(fields.Field):
    def _deserialize(self, raw: dict, attr, data, **kwargs):
        validated_dict = PointSchema().load(raw)
        return Point(**validated_dict)


PointType = NewType('Point', Point, PointField)


class CurveField(fields.Field):
    def _deserialize(self, raw: dict, attr, data, **kwargs):
        """
        This isn't currently using a Marshmallow schema to validate, perhaps there is a way of doing that elegantly.
        """
        points = []
        for point_config in raw:
            points.append(Point(x=float(point_config["x"]), y=float(point_config["y"])))

        return Curve(points)


CurveType = NewType('Curve', Curve, CurveField)
