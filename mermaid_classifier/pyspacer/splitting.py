"""
Image-grouped stratified train/ref/val split.

Every image lands whole in one set, so no set shares imagery with
another. PySpacer's own point-level split cannot do this, and it caps ref
at TRAINING_BATCH_LABEL_COUNT points; this split applies the ratios as
given.
"""

import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Callable

from spacer.data_classes import Annotation, DataLocation, ImageLabels, LabelId
from spacer.messages import TrainingTaskLabels

logger = logging.getLogger(__name__)

# A class in fewer images than this cannot reach all three sets.
MIN_IMAGES_PER_CLASS = 3

# Also the tie-break order: a rare class's first images seed val, then ref.
_SET_NAMES = ("val", "ref", "train")


def _seeded_digest(seed: int, key: str) -> bytes:
    # Python's hash() is salted per process; blake2b is not.
    return hashlib.blake2b(f"{seed}\x00{key}".encode(), digest_size=8).digest()


def split_by_image(
    labels: ImageLabels,
    ratios: tuple[float, float],
    seed: int,
    *,
    stable_key: Callable[[DataLocation], str] | None = None,
) -> TrainingTaskLabels:
    """
    Split `labels` into train/ref/val by whole image, stratified by class.

    `ratios` is (ref, val); train takes the rest. Classes present in fewer
    than MIN_IMAGES_PER_CLASS images are dropped first, and logged.

    Deterministic greedy iterative stratification. Each image is ranked by
    the total point count of its rarest class, ascending, with a seeded
    hash of `stable_key(location)` (default: the location's key) as the
    tie-break. In that order, each image goes to the set whose share of
    its rarest class is furthest below target, i.e. the set minimizing
    assigned / ratio for that class. A class's first images therefore seed
    val, ref and train before any set gets a second one. A repair pass then
    moves or swaps images so a class that is never its images' rarest class
    still reaches all three sets.

    Pass a `stable_key` when locations embed a per-run path (such as a
    temp download dir), so the split does not depend on it.

    Complexity: O(P + I log I) time and O(P) memory, for P points over I
    images. Each point is visited a constant number of times; only the
    image ordering is a sort.
    """
    ref_ratio, val_ratio = ratios
    fractions = {"val": val_ratio, "ref": ref_ratio, "train": 1.0 - ref_ratio - val_ratio}
    if min(fractions.values()) <= 0:
        raise ValueError(
            f"Split ratios {ratios} must give each of train, ref and val a positive share."
        )
    key_of = stable_key if stable_key is not None else (lambda loc: loc.key)

    images = [(loc, labels[loc]) for loc in labels.keys()]  # noqa: SIM118 -- ImageLabels is not a dict

    images_per_class: Counter[LabelId] = Counter()
    for _, annotations in images:
        images_per_class.update({label for _, _, label in annotations})
    dropped = {c for c, n in images_per_class.items() if n < MIN_IMAGES_PER_CLASS}
    if dropped:
        logger.info(
            "Image split drops %d class(es) present in fewer than %d images"
            " (class: images, points): %s",
            len(dropped),
            MIN_IMAGES_PER_CLASS,
            ", ".join(
                f"{c}: {images_per_class[c]}, {labels.label_count_per_class[c]}"
                for c in sorted(dropped, key=str)
            ),
        )
        kept_images: list[tuple[DataLocation, list[Annotation]]] = []
        for loc, annotations in images:
            kept = [a for a in annotations if a[2] not in dropped]
            if kept:
                kept_images.append((loc, kept))
        images = kept_images

    class_points: Counter[LabelId] = Counter()
    for _, annotations in images:
        class_points.update(label for _, _, label in annotations)

    ranked: list[tuple[int, bytes, str, LabelId, DataLocation, list[Annotation]]] = []
    for loc, annotations in images:
        rarest = min(
            {label for _, _, label in annotations},
            key=lambda label: (class_points[label], str(label)),
        )
        key = key_of(loc)
        ranked.append(
            (class_points[rarest], _seeded_digest(seed, key), key, rarest, loc, annotations)
        )
    ranked.sort(key=lambda entry: entry[:3])

    assigned: dict[str, Counter[LabelId]] = {name: Counter() for name in _SET_NAMES}
    sets: dict[str, dict[DataLocation, list[Annotation]]] = {name: {} for name in _SET_NAMES}
    for _, _, _, rarest, loc, annotations in ranked:
        target = min(_SET_NAMES, key=lambda name: assigned[name][rarest] / fractions[name])
        sets[target][loc] = annotations
        assigned[target].update(label for _, _, label in annotations)

    _repair_missing_classes(sets, {loc: digest for _, digest, _, _, loc, _ in ranked})

    return TrainingTaskLabels(
        train=ImageLabels(sets["train"]),
        ref=ImageLabels(sets["ref"]),
        val=ImageLabels(sets["val"]),
    )


def _repair_missing_classes(
    sets: dict[str, dict[DataLocation, list[Annotation]]],
    digests: dict[DataLocation, bytes],
) -> None:
    """
    Move images, in place, until every class is in every set it can reach.

    The greedy pass places an image by its rarest class alone, so a class
    that is never the rarest in any of its images can miss a set, and
    PySpacer then drops it from all three. For each missing (set, class)
    pair, an image holding the class moves in from the set with the most
    images of it, lowest seeded digest first. A move never takes the last
    image of any class out of its set. When every candidate would, the
    candidate is swapped with an image of the missing set that restores
    what it takes away.

    Each move or swap adds one (set, class) pair and removes none, so the
    loop ends. Pairs still missing at the end are logged as a warning.
    """
    labels_of = {
        loc: frozenset(label for _, _, label in annotations)
        for members in sets.values()
        for loc, annotations in members.items()
    }
    home = {loc: name for name, members in sets.items() for loc in members}
    carriers: dict[str, Counter[LabelId]] = {name: Counter() for name in _SET_NAMES}
    for loc, name in home.items():
        carriers[name].update(labels_of[loc])
    holders: dict[LabelId, list[DataLocation]] = defaultdict(list)
    for loc in sorted(labels_of, key=lambda loc: (digests[loc], loc.key)):
        for label in labels_of[loc]:
            holders[label].append(loc)
    classes = sorted(holders, key=str)

    def lost_on_leaving(loc: DataLocation, arriving: frozenset[LabelId]) -> set[LabelId]:
        name = home[loc]
        return {
            label
            for label in labels_of[loc]
            if carriers[name][label] == 1 and label not in arriving
        }

    def move(loc: DataLocation, target: str) -> None:
        source = home[loc]
        sets[target][loc] = sets[source].pop(loc)
        carriers[source].subtract(labels_of[loc])
        carriers[target].update(labels_of[loc])
        home[loc] = target

    def fill(label: LabelId, target: str) -> bool:
        donors = sorted(
            (name for name in _SET_NAMES if name != target and carriers[name][label]),
            key=lambda name: -carriers[name][label],
        )
        candidates = [loc for donor in donors for loc in holders[label] if home[loc] == donor]
        for loc in candidates:
            if not lost_on_leaving(loc, frozenset()):
                move(loc, target)
                return True
        for loc in candidates:
            lost = lost_on_leaving(loc, frozenset())
            if label in lost:
                continue
            donor = home[loc]
            for partner in holders[min(lost, key=str)]:
                if (
                    home[partner] == target
                    and lost <= labels_of[partner]
                    and not lost_on_leaving(partner, labels_of[loc])
                ):
                    move(loc, target)
                    move(partner, donor)
                    return True
        return False

    progress = True
    while progress:
        progress = False
        for label in classes:
            for target in _SET_NAMES:
                if not carriers[target][label] and fill(label, target):
                    progress = True

    missing = [
        f"{label}: {name}" for label in classes for name in _SET_NAMES if not carriers[name][label]
    ]
    if missing:
        logger.warning(
            "Image split could not place %d class(es) in every set, so PySpacer"
            " drops them (class: missing set): %s",
            len(missing),
            ", ".join(missing),
        )
