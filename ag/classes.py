"""BDD100K / Cityscapes 19-class train-id scheme (identical ordering in both datasets)."""

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
IGNORE = 255

ROAD, SIDEWALK, BUILDING, WALL, FENCE = 0, 1, 2, 3, 4
SKY = 10
PERSON, RIDER = 11, 12
CAR, TRUCK, BUS, TRAIN, MOTORCYCLE, BICYCLE = 13, 14, 15, 16, 17, 18
VEHICLES = (CAR, TRUCK, BUS, MOTORCYCLE, BICYCLE)

# Cityscapes colour palette, used for overlays in the platform.
PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
