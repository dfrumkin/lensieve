from lensieve.data.orientation import AspectRatio, OrientationKind
from lensieve.image import IMAGE_FORMATS

ASPECT_RATIO_VALUES = "\n".join(f"  - {ratio.value}" for ratio in AspectRatio)

ORIENTATION_KIND_VALUES = "\n".join(f"  - {kind.value}" for kind in OrientationKind)

IMAGE_FORMAT_VALUES = "\n".join(f"  - {fmt}" for fmt in sorted(IMAGE_FORMATS))
