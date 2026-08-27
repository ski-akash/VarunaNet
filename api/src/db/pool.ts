import pg from "pg";

// One place that constructs the Postgres connection pool, matching
// redisConnection.ts's role for Redis -- callers depend on a `pg.Pool`,
// never on `process.env` or the `pg` package directly.
export function createPgPool(databaseUrl: string): pg.Pool {
  return new pg.Pool({ connectionString: databaseUrl });
}
