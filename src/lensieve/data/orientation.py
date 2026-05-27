from enum import StrEnum

from lensieve.image import ExifOrientation

SWAPS_WIDTH_HEIGHT: frozenset[ExifOrientation] = frozenset(
    {
        ExifOrientation.MIRRORED_HORIZONTAL_ROTATED_90_CCW,
        ExifOrientation.ROTATED_90_CW,
        ExifOrientation.MIRRORED_HORIZONTAL_ROTATED_90_CW,
        ExifOrientation.ROTATED_90_CCW,
    }
)

ASPECT_RATIO_EPSILON = 0.03


class AspectRatio(StrEnum):
    RATIO_1_1 = "1:1"

    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"

    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"

    RATIO_3_2 = "3:2"
    RATIO_2_3 = "2:3"

    LANDSCAPE_OTHER = "landscape_other"
    PORTRAIT_OTHER = "portrait_other"


ASPECT_RATIO_BUCKETS: tuple[tuple[float, AspectRatio, AspectRatio], ...] = (
    (
        16 / 9,
        AspectRatio.RATIO_16_9,
        AspectRatio.RATIO_9_16,
    ),
    (
        4 / 3,
        AspectRatio.RATIO_4_3,
        AspectRatio.RATIO_3_4,
    ),
    (
        3 / 2,
        AspectRatio.RATIO_3_2,
        AspectRatio.RATIO_2_3,
    ),
)


class OrientationKind(StrEnum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"
