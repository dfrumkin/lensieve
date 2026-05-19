from dataclasses import dataclass

import numpy as np

from lensieve.retrieval.schema import ImageHit, SearchResult


@dataclass(frozen=True, slots=True)
class ImageGroupItem:
    hit: ImageHit
    similarity_to_representative: float  # 1.0 for representative


@dataclass(frozen=True, slots=True)
class ImageGroup:
    representative: ImageHit
    items: tuple[ImageGroupItem, ...]


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

        sha256 = hits[i].sha256
        group_indices = [i]

        for j in range(i + 1, n_hits):
            if not assigned[j] and (hits[j].sha256 == sha256 or sim[i, j] >= threshold):
                assigned[j] = True
                group_indices.append(j)

        # Representative first, then descending similarity to representative
        group_indices.sort(
            key=lambda idx: 1.0 if idx == i else np.minimum(sim[i, idx], 1.0),
            reverse=True,
        )

        items = tuple(
            ImageGroupItem(
                hit=hits[idx],
                similarity_to_representative=min(float(sim[i, idx]), 1.0),
            )
            for idx in group_indices
        )

        groups.append(
            ImageGroup(
                representative=hits[i],
                items=items,
            )
        )

    return groups
