class PGVectorStore:
    def __init__(self, conn):
        self.conn = conn

    def similarity_search_with_score(self, query_vector, k=5):
        """
        Returns the top k similar vectors to the query vector.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, vector, ts_rank_cd(vector, query) AS score FROM {self.table} ORDER BY vector <-> query LIMIT {k}",
                {"query": query_vector},
            )
            results = cur.fetchall()
            return results

    def similarity_search(self, query_vector, k=5):
        """
        Returns the top k similar vectors to the query vector.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, vector FROM {self.table} ORDER BY vector <-> query LIMIT {k}",
                {"query": query_vector},
            )
            results = cur.fetchall()
            return results
