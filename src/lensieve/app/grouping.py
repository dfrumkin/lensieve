from dataclasses import dataclass

import numpy as np

from lensieve.retrieval.schema import ImageHit, SearchResult


@dataclass(frozen=True, slots=True)
class ImageGroup:
    representative: ImageHit
    items: tuple[ImageHit, ...]


def group_hits_by_similarity(
    result: SearchResult,
    threshold: float,
) -> list[ImageGroup]:
    hits = result.hits
    n_hits = len(hits)

    if n_hits == 0:
        return []

    sim = result.similarity_matrix
    assigned = np.zeros(n_hits, dtype=bool)
    groups: list[ImageGroup] = []

    for i in range(n_hits):
        if assigned[i]:
            continue

        # start new group with representative i
        group_indices = [i]
        sha256 = hits[i].sha256

        # add all similar, unassigned items
        for j in range(i + 1, n_hits):
            if not assigned[j] and (hits[j].sha256 == sha256 or sim[i, j] >= threshold):
                assigned[j] = True
                group_indices.append(j)

        # sort group items by score (optional but nice)
        group_indices.sort(key=lambda idx: hits[idx].score, reverse=True)

        items = tuple(hits[idx] for idx in group_indices)
        groups.append(
            ImageGroup(
                representative=items[0],
                items=items,
            )
        )

    return groups
