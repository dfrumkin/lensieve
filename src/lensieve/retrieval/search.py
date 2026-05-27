from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa

from lensieve.data.data_store import DataStore
from lensieve.data.utils import duck_table_name, sql_ident
from lensieve.image import load_image
from lensieve.models.clip_like_embedder import ClipLikeEmbedder
from lensieve.models.model_manager import ModelManager, ModelRole
from lensieve.models.vision_embedder import VisionEmbedder
from lensieve.names import DISTANCE_COL
from lensieve.names import BaseField as BF
from lensieve.names import EmbeddingField as EF
from lensieve.names import ImageField as IF
from lensieve.names import TableName as TN
from lensieve.retrieval.schema import ImageHit, SearchArgs, SearchResult, Similarity


def search_images(args: SearchArgs, model_manager: ModelManager, data_store: DataStore) -> SearchResult:
    matches = find_matches(args, model_manager, data_store)

    if matches:
        # Here, we introduce duplicate images (same sha, different path) if there were any
        matches = add_from_table(data_store, matches, TN.IMAGES, [IF.PATH])

        hits = get_hits(matches=matches, root=data_store.root)
        similarity = calc_similarity_matrix(model_manager, data_store, matches, args.text_query is not None)
    else:
        hits = ()
        similarity = np.empty((0, 0), dtype=np.float32)

    return SearchResult(hits=hits, similarity_matrix=similarity)


def find_matches(args: SearchArgs, model_manager: ModelManager, data_store: DataStore) -> pa.Table:
    if args.text_query:
        model = ClipLikeEmbedder(manager=model_manager)
        query = " ".join(args.text_query.strip().split())
        # Normalize short visual queries into a caption-like form.
        query = f"a photo of {query.lower()}"
        query_vector = model.run(texts=[query])[0]
    else:
        model = VisionEmbedder(manager=model_manager)
        if args.image_query_path:
            image_path = args.image_query_path
        else:
            path_col = sql_ident(IF.PATH)
            hash_col = sql_ident(IF.SHA256)
            with data_store.connect_duckdb_for_lance() as con:
                row = con.execute(
                    f"""
                    SELECT {path_col}
                    FROM {duck_table_name(TN.IMAGES)}
                    WHERE {hash_col} = ?
                    LIMIT 1
                    """,
                    [args.image_query_sha256],
                ).fetchone()

            if row is None:
                raise ValueError(f"No image found for SHA: {args.image_query_sha256}")
            image_path = Path(data_store.root / row[0])

        image = load_image(image_path)
        query_vector = model.run(images=[image])[0]

    table_name = TN.embeddings(model.model_name)
    table = data_store.lancedb.open_table(table_name)
    query = (
        table.search(query_vector, vector_column_name=EF.VECTOR)
        .metric("cosine")  # type: ignore[attr-defined]
        .select([EF.SHA256, EF.VECTOR, DISTANCE_COL])
    )

    clauses = []

    if args.date_start is not None:
        start_dt = datetime.combine(args.date_start, time.min)
        clauses.append(f"{EF.DATE_TAKEN} >= TIMESTAMP '{start_dt.isoformat(sep=' ')}'")

    if args.date_end is not None:
        end_dt = datetime.combine(args.date_end + timedelta(days=1), time.min)
        clauses.append(f"{EF.DATE_TAKEN} < TIMESTAMP '{end_dt.isoformat(sep=' ')}'")

    if clauses:
        where = " AND ".join(clauses)
        query = query.where(where, prefilter=True)

    return query.limit(args.max_results).to_arrow()


def add_from_table(
    data_store: DataStore,
    matches: pa.Table,
    table_name: str,
    columns: list[str],
) -> pa.Table:
    if matches.num_rows == 0 or not columns:
        return matches

    sha_col = sql_ident(BF.SHA256)
    dist_col = sql_ident(DISTANCE_COL)
    table = duck_table_name(table_name)

    selected_cols = ", ".join(["m.*"] + [f"v.{sql_ident(col)}" for col in columns])

    with data_store.connect_duckdb_for_lance() as con:
        con.register("matches", matches)

        return con.execute(
            f"""
            SELECT {selected_cols}
            FROM matches AS m
            LEFT JOIN {table} AS v
            ON m.{sha_col} = v.{sha_col}
            ORDER BY m.{dist_col}
            """
        ).fetch_arrow_table()


def get_hits(matches: pa.Table, root: Path) -> tuple[ImageHit, ...]:
    paths = matches[IF.PATH].to_pylist()
    shas = matches[BF.SHA256].to_pylist()
    distances = matches[DISTANCE_COL].to_pylist()

    return tuple(
        ImageHit(
            path=root / path,
            sha256=str(sha),
            score=float(1.0 - distance),
        )
        for path, sha, distance in zip(paths, shas, distances, strict=True)
    )


def calc_similarity_matrix(
    model_manager: ModelManager, data_store: DataStore, matches: pa.Table, text_query: bool
) -> Similarity:
    if text_query:
        matches = add_from_table(
            data_store=data_store,
            matches=matches.drop([EF.VECTOR]),
            table_name=TN.embeddings(model_manager.get_model_name(ModelRole.VISION)),
            columns=[EF.VECTOR],
        )

    emb = np.asarray(matches[EF.VECTOR].to_pylist(), dtype=np.float32)

    return emb @ emb.T
