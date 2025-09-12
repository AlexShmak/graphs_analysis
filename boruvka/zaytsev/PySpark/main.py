from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, first, collect_list, explode
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import itertools
from pyspark import StorageLevel

class BoruvkasMST:
    def __init__(self):
        # https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/SparkSession.Builder.html
        self.spark = SparkSession.builder \
            .appName("BoruvkaMST") \
            .master("local[*]") \
            .getOrCreate()
        
        self.sc = self.spark.sparkContext # connect with Java SparkContext via TCP

    def find_mst(self, edges_data: list[tuple[int, int, float]], vertices_count: int):
        """
        Find Minimum Spanning Tree using Boruvka's algorithm
        
        Args:
            edges_data: List of tuples (u, v, weight) representing edges
            num_vertices: Number of vertices in the graph
            
        Returns:
            List of edges in the MST
        """
        # Define schema for edges
        edge_schema = StructType([
            StructField("u", IntegerType(), True),
            StructField("v", IntegerType(), True),
            StructField("weight", DoubleType(), True)
        ])
        
        # Create DataFrame from edges
        edges_df = self.spark.createDataFrame(edges_data, edge_schema).persist(StorageLevel.MEMORY_ONLY)
        
        # Initialize Union-Find structure
        components = list(range(vertices_count))
        mst_edges = []
        
        while True:
            # Create component mapping DataFrame
            # component_data = find root for every vertex (vertex, root)
            component_data = [(i, self._find_root(components, i)) for i in range(vertices_count)]
            component_schema = StructType([
                StructField("vertex", IntegerType(), True),
                StructField("component", IntegerType(), True)
            ])
            component_df = self.spark.createDataFrame(component_data, component_schema).persist(StorageLevel.MEMORY_ONLY)
            
            # Add component information to edges
            edges_with_comp = edges_df.alias("e") \
                .join(component_df.alias("c1"), col("e.u") == col("c1.vertex")) \
                .join(component_df.alias("c2"), col("e.v") == col("c2.vertex")) \
                .select(
                    col("e.u"), col("e.v"), col("e.weight"),
                    col("c1.component").alias("comp_u"),
                    col("c2.component").alias("comp_v")
                ) \
                .filter(col("comp_u") != col("comp_v")).persist(StorageLevel.MEMORY_ONLY)  # Only edges between different components
            
            # Check if we have any edges left (algorithm termination condition)
            if edges_with_comp.count() == 0:
                break
            
            # Find minimum edge for each component
            # We need to consider edges from both directions for each component
            edges_from_u = edges_with_comp.select(
                col("comp_u").alias("component"),
                col("u"), col("v"), col("weight")
            ).persist(StorageLevel.MEMORY_ONLY)
            
            edges_from_v = edges_with_comp.select(
                col("comp_v").alias("component"),
                col("v").alias("u"), col("u").alias("v"), col("weight")
            ).persist(StorageLevel.MEMORY_ONLY)

            all_component_edges = edges_from_u.union(edges_from_v)

            # Find minimum weight edge for each component
            min_edges = all_component_edges \
                .groupBy("component") \
                .agg(spark_min("weight").alias("min_weight")) \
                .join(all_component_edges, ["component"]) \
                .filter(col("weight") == col("min_weight")) \
                .groupBy("component") \
                .agg(
                    first("u").alias("u"),
                    first("v").alias("v"),
                    first("weight").alias("weight")
                ).persist(StorageLevel.MEMORY_ONLY)
            
            # Collect minimum edges
            selected_edges = min_edges.collect()
            
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

if __name__ == "__main__":
    edges = [(0, 1, 3.0), (0, 4, 7.0), (1,4,5.0), (1,2,1.0), (2,4,6.0), (2,3,10.0), (3,4,8.0)]
    vertices_count = 5
    mst = BoruvkasMST()

    res = mst.find_mst(edges, vertices_count)

    print("Result:", res)