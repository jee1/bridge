# Elasticsearch Configuration

Bridge supports Elasticsearch connectivity via the `ElasticsearchConnector`. Configure it with either a full connection URL or separate host credentials:

- `BRIDGE_ELASTICSEARCH_URL`: Optional. Full URL including scheme and port (for example, `https://es.example.com:9243`). When provided it overrides the host/port settings and automatically enables TLS when the scheme is `https`.
- `BRIDGE_ELASTICSEARCH_HOST`: Hostname of the cluster. Defaults to `localhost` when the URL is not set.
- `BRIDGE_ELASTICSEARCH_PORT`: Port number for the cluster. Defaults to `9200`.
- `BRIDGE_ELASTICSEARCH_USE_SSL`: Set to `true` to enable TLS when using host/port values.
- `BRIDGE_ELASTICSEARCH_USERNAME` / `BRIDGE_ELASTICSEARCH_PASSWORD`: Optional basic authentication credentials.

Add the variables to your `.env` file or export them before starting the MCP server. The connector validates the configuration and logs the resolved URL so you can confirm what will be used at runtime.
