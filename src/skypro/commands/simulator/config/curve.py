from marshmallow import Schema, fields
from marshmallow_dataclass import NewType

from skypro.commands.simulator.cartesian import Point, Curve


"""
This handles parsing of JSON into a Curve and Point types 
"""


class PointSchema(Schema):
    x = fields.Float
    y = fields.Float
    # TODO: somehow these aren't being validated - Infinity is coming through as a string


class PointField(fields.Field):
    def _deserialize(self, raw: dict, attr, data, **kwargs):
        validated_dict = PointSchema().load(raw)
        return Point(**validated_dict)


PointType = NewType('Point', Point, PointField)


class CurveField(fields.Field):
    def _deserialize(self, raw: dict, attr, data, **kwargs):
        """
        This isn't current using a Marshmallow schema to validate, perhaps there is a way of doing that elegantly.
        """
        points = []
        for point_config in raw:
            points.append(Point(x=point_config["x"], y=point_config["y"]))

        return Curve(points)


CurveType = NewType('Curve', Curve, CurveField)
