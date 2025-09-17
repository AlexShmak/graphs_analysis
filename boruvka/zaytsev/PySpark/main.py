from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, first
from pyspark.sql.types import StructType, StructField, IntegerType
from pathlib import Path
import logging
from pyspark import StorageLevel
from pyspark.sql.functions import broadcast
import time
import math
import os
import sys
import gc

logger = logging.getLogger("experiments")
logger.setLevel(logging.INFO)
hdl = logging.FileHandler("results.log")
formatter = logging.Formatter("%(message)s")
hdl.setFormatter(formatter)
logger.addHandler(hdl)

os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx32g"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--conf spark.driver.extraJavaOptions=-Dlog4j2.formatMsgNoLookups=true pyspark-shell"
)


class BoruvkasMST:
    def __init__(self):
        # https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/SparkSession.Builder.html
        self.spark = (
            SparkSession.builder.appName("BoruvkaMST")
            .master("local[*]")
            .config("spark.driver.memory", "32g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("ERROR")

    def find_mst(self, edges_data: list[tuple[int, int, int]], vertices_count: int):
        """Run MST.

        Args:
            edges_data (list[tuple[int, int, int]]): Edges data. Edges should always be in [0, vertices_count-1] range.
            vertices_count (int): vertices count.
        """
        # Define schema for edges
        edge_schema = StructType(
            [
                StructField("u", IntegerType(), True),
                StructField("v", IntegerType(), True),
                StructField("weight", IntegerType(), True),
            ]
        )

        # Create DataFrame from edges
        edges_df = self.spark.createDataFrame(edges_data, edge_schema).persist(
            StorageLevel.MEMORY_ONLY
        )

        # Initialize Union-Find structure
        components = list(range(vertices_count))
        mst_edges = []

        component_schema = StructType(
            [
                StructField("vertex", IntegerType(), True),
                StructField("component", IntegerType(), True),
            ]
        )

        while True:
            # Create component mapping DataFrame
            # component_data = find root for every vertex (vertex, root)
            component_data = [
                (i, self._find_root(components, i)) for i in range(vertices_count)
            ]
            component_df = self.spark.createDataFrame(
                component_data, component_schema
            ).persist(StorageLevel.MEMORY_ONLY)

            # Add component information to edges
            edges_with_comp = (
                edges_df.alias("e")
                .join(
                    broadcast(component_df.alias("c1")), col("e.u") == col("c1.vertex")
                )
                .join(
                    broadcast(component_df.alias("c2")), col("e.v") == col("c2.vertex")
                )
                .select(
                    col("e.u"),
                    col("e.v"),
                    col("e.weight"),
                    col("c1.component").alias("comp_u"),
                    col("c2.component").alias("comp_v"),
                )
                .filter(col("comp_u") != col("comp_v"))
                .persist(StorageLevel.MEMORY_ONLY)
            )  # Only edges between different components

            component_df.unpersist(blocking=True)

            # Check if we have any edges left (algorithm termination condition)
            if edges_with_comp.count() == 0:
                break

            # Find minimum edge for each component
            # We need to consider edges from both directions for each component
            edges_from_u = edges_with_comp.select(
                col("comp_u").alias("component"), col("u"), col("v"), col("weight")
            ).persist(StorageLevel.MEMORY_ONLY)

            edges_from_v = edges_with_comp.select(
                col("comp_v").alias("component"),
                col("v").alias("u"),
                col("u").alias("v"),
                col("weight"),
            ).persist(StorageLevel.MEMORY_ONLY)

            edges_with_comp.unpersist(blocking=True)

            all_component_edges = edges_from_u.union(edges_from_v)

            edges_from_u.unpersist(blocking=True)
            edges_from_v.unpersist(blocking=True)

            # Find minimum weight edge for each component
            min_edges = (
                all_component_edges.groupBy("component")
                .agg(spark_min("weight").alias("min_weight"))
                .join(all_component_edges, ["component"])
                .filter(col("weight") == col("min_weight"))
                .groupBy("component")
                .agg(
                    first("u").alias("u"),
                    first("v").alias("v"),
                    first("weight").alias("weight"),
                )
                .persist(StorageLevel.MEMORY_ONLY)
            )

            all_component_edges.unpersist(blocking=True)

            # Collect minimum edges
            selected_edges = min_edges.collect()
            min_edges.unpersist(blocking=True)

            if not selected_edges:
                break

            # Process selected edges and update Union-Find
            edges_to_add = set()
            for row in selected_edges:
                u, v, weight = row.u, row.v, row.weight
                root_u = self._find_root(components, u)
                root_v = self._find_root(components, v)

                if root_u != root_v:
                    # Create canonical edge representation (smaller vertex first)
                    edge = (min(u, v), max(u, v), weight)
                    edges_to_add.add(edge)
                    # Union the components
                    self._union(components, u, v)

            # Add unique edges to MST
            mst_edges.extend(list(edges_to_add))

            # Remove edges that are no longer needed (both endpoints in same component)
            # This is handled in the next iteration by the component filtering

        return mst_edges

    def clean(self):
        self.spark.catalog.clearCache()
        self.spark.sparkContext._jsc.getPersistentRDDs().clear()
        gc.collect()

    def stop(self):
        self.spark.stop()

    def _find_root(self, parent, x):
        """Find root of element x with path compression"""
        if parent[x] != x:
            parent[x] = self._find_root(parent, parent[x])
        return parent[x]

    def _union(self, parent, x, y):
        """Union two sets"""
        root_x = self._find_root(parent, x)
        root_y = self._find_root(parent, y)
        if root_x != root_y:
            parent[root_y] = root_x


def read_dimacs_file(file: Path) -> tuple[int, list[tuple[int, int, int]]]:
    edges = set()
    vertices_count = 1

    with file.open() as f:
        for line in f:
            if line.startswith("p"):
                _, _, vertices_count, _ = line.strip().split()
                vertices_count = int(vertices_count)
            if line.startswith("a"):
                _, u, v, w = line.strip().split()
                u, v, w = int(u), int(v), int(w)
                edges.add((min(u, v) - 1, max(u, v) - 1, w))

    return vertices_count, list(edges)


def warm_up(mst: BoruvkasMST):
    print("Starting warm-up...")
    vertices_count, edges = read_dimacs_file(Path("../dataset-dimacs/NY.gr"))
    mst.find_mst(edges, vertices_count)
    print("Warm-up completed successfully.")


def run_tests(mst: BoruvkasMST):
    # test 1
    edges = [
        (0, 1, 10),
        (1, 2, 10),
        (2, 3, 10),
        (3, 0, 10),
        (1, 3, 1),
        (0, 2, 1),
        (1, 0, 10),
        (2, 1, 10),
        (3, 2, 10),
        (0, 3, 10),
        (3, 1, 1),
        (2, 0, 1),
    ]
    vertices_count = 4

    res = mst.find_mst(edges, vertices_count)

    assert res == [(0, 2, 1), (1, 3, 1), (1, 2, 10)], res

    # test 2
    vertices_count, edges = read_dimacs_file(Path("../graph-utils/test-graph.gr"))
    assert vertices_count == 5
    assert len(edges) == 7, edges

    res = mst.find_mst(edges, vertices_count)

    assert res == [(1, 4, 8), (1, 2, 4), (0, 3, 3), (0, 1, 5)], res


REPEATS = 100


def run_experiment(mst: BoruvkasMST, filename: Path):
    vertices_count, edges = read_dimacs_file(filename)

    time_array = []

    for _ in range(REPEATS):
        start = time.monotonic()
        mst.find_mst(edges, vertices_count)
        end = time.monotonic()
        time_array.append(end - start)
        mst.clean()
    
    filename.unlink()

    # --- Statistics ---
    mean = sum(time_array) / len(time_array)
    variance = sum((t - mean) ** 2 for t in time_array) / (len(time_array) - 1)
    stddev = math.sqrt(variance)
    z = 1.96
    margin = z * (stddev / math.sqrt(len(time_array)))

    logger.info(f"Mean = {mean:.6f}s")
    logger.info(f"StdDev = {stddev:.6f}s")
    logger.info(f"95% CI = ±{margin:.6f}s")


def run_experiments_on_dataset(mst: BoruvkasMST, folder: Path):
    if not folder.is_dir():
        print(f"opendir failed: {folder} is not a directory.")
        return

    for entry in folder.iterdir():
        if not entry.is_file():
            continue

        print(f"Run experiment for: {entry.name}")
        logger.info(f"Run experiment for: {entry.name}")

        run_experiment(mst, entry)


if __name__ == "__main__":
    mst = BoruvkasMST()
    warm_up(mst)
    run_tests(mst)

    try:
        run_experiments_on_dataset(mst, Path("../dataset-my/"))
        run_experiments_on_dataset(mst, Path("../dataset-my/ext-dataset-exp1"))
        run_experiments_on_dataset(mst, Path("../dataset-my/ext-dataset-exp2"))
    except Exception as e:
        logger.error(f"Experiment error: {e}.")
        sys.exit(1)