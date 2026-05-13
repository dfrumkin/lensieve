from pathlib import Path

import numpy as np
import pandas as pd

from lensieve.consts import DISTANCE_COL
from lensieve.consts import BaseField as BF
from lensieve.consts import EmbeddingField as EF
from lensieve.consts import ImageField as IF
from lensieve.consts import TableName as TN
from lensieve.data_store import DataStore, duck_table_name, sql_ident
from lensieve.image import load_image
from lensieve.models.clip_like_embedder import ClipLikeEmbedder
from lensieve.models.model_manager import ModelKind, ModelManager
from lensieve.models.vision_embedder import VisionEmbedder
from lensieve.retrieval.schema import ImageHit, SearchArgs, SearchResult


def search_images(args: SearchArgs, model_manager: ModelManager, data_store: DataStore) -> SearchResult:
    matches = find_matches(args, model_manager, data_store)
    # Here, we introduce duplicate images if there were any
    matches = add_from_table(data_store, matches, TN.IMAGES, [IF.PATH])

    hits = [
        ImageHit(
            path=str(data_store.root / row[IF.PATH]),
            sha256=str(row[BF.SHA256]),
            score=float(1.0 - row[DISTANCE_COL]),
        )
        for _, row in matches.iterrows()
    ]
    similarity = calc_similarity_matrix(args, model_manager, data_store, matches)

    return SearchResult(hits=hits, similarity_matrix=similarity)


def find_matches(args: SearchArgs, model_manager: ModelManager, data_store: DataStore) -> pd.DataFrame:
    if args.text_query:
        model = ClipLikeEmbedder(manager=model_manager)
        query_vector = model.run(texts=[args.text_query])[0]
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
    date_col = sql_ident(EF.DATE_TAKEN)

    if args.date_start is not None:
        clauses.append(f"{date_col} >= TIMESTAMP '{args.date_start.isoformat()}'")

    if args.date_end is not None:
        clauses.append(f"{date_col} <= TIMESTAMP '{args.date_end.isoformat()}'")

    if clauses:
        where = " AND ".join(clauses)
        query = query.where(where, prefilter=True)

    return query.limit(args.max_results).to_pandas()


def add_from_table(
    data_store: DataStore,
    matches: pd.DataFrame,
    table_name: str,
    columns: list[str],
) -> pd.DataFrame:
    if matches.empty or not columns:
        return matches.copy()

    sha_col = sql_ident(BF.SHA256)
    table = duck_table_name(table_name)

    select_cols = ", ".join([f"v.{sha_col}"] + [f"v.{sql_ident(col)}" for col in columns])

    shas = matches[BF.SHA256].drop_duplicates().tolist()

    placeholders = ", ".join(["?"] * len(shas))

    with data_store.connect_duckdb_for_lance() as con:
        fetched = con.execute(
            f"""
            SELECT {select_cols}
            FROM {table} AS v
            WHERE v.{sha_col} IN ({placeholders})
            """,
            shas,
        ).fetchdf()

    return matches.merge(
        fetched,
        on=BF.SHA256,
        how="left",
        sort=False,
    )


def calc_similarity_matrix(
    args: SearchArgs, model_manager: ModelManager, data_store: DataStore, matches: pd.DataFrame
) -> list[list[float]]:
    if matches.empty:
        return []

    if args.text_query:
        emb_series = add_from_table(
            data_store=data_store,
            matches=matches.drop(columns=[EF.VECTOR]),
            table_name=TN.embeddings(model_manager.get_model_name(ModelKind.VISION)),
            columns=[EF.VECTOR],
        )[EF.VECTOR]
    else:
        emb_series = matches[EF.VECTOR]
    emb = np.stack([np.asarray(x) for x in emb_series]).astype(np.float32)
    return (emb @ emb.T).tolist()
