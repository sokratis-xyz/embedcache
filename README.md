# EmbedCache : high-performance rust webpage embedding service with caching

EmbedCache is a high-performance text embedding service with caching capabilities, built in Rust. It provides a REST API for text chunking and embedding generation using various state-of-the-art models.

## Features

- **Multiple Embedding Models**: Support for 20+ embedding models including BGE, MiniLM, Nomic, and multilingual models
- **Flexible Chunking Strategies**: 
  - Word-based chunking
  - LLM-based concept chunking (planned)
  - LLM-based introspection chunking (planned)
- **Caching**: SQLite-based caching with configurable journal modes
- **API Documentation**: Built-in Swagger UI, Redoc, and RapiDoc interfaces
- **Environment Configuration**: Flexible configuration through environment variables

## Quick Start

### Prerequisites

- Rust (latest stable version)
- SQLite

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/embedcache.git
cd embedcache
```

2. Create a `.env` file with your configuration:
```bash
SERVER_HOST=127.0.0.1
SERVER_PORT=8081
DB_PATH=cache.db
DB_JOURNAL_MODE=wal
ENABLED_MODELS=BGESmallENV15,AllMiniLML6V2
```

3. Build and run:
```bash
cargo build --release
cargo run --release
```

The server will start at `http://127.0.0.1:8081` (or your configured host/port).

## API Endpoints

### `POST /v1/embed`
Generate embeddings for a list of text strings.

Request body:
```json
{
  "text": ["your text here", "another text"],
  "config": {
    "chunking_type": "words",
    "chunking_size": 512,
    "embedding_model": "BGESmallENV15"
  }
}
```

### `POST /v1/process`
Process a URL by extracting content, chunking, and generating embeddings.

Request body:
```json
{
  "url": "https://example.com",
  "config": {
    "chunking_type": "words",
    "chunking_size": 512,
    "embedding_model": "BGESmallENV15"
  }
}
```

### `GET /v1/params`
List supported chunking types and embedding models.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| SERVER_HOST | 127.0.0.1 | Server host address |
| SERVER_PORT | 8081 | Server port |
| DB_PATH | cache.db | SQLite database path |
| DB_JOURNAL_MODE | wal | SQLite journal mode (wal/truncate/persist) |
| ENABLED_MODELS | AllMiniLML6V2 | Comma-separated list of enabled models |

## Supported Models

The service supports various embedding models including:
- BGE models (Small, Base, Large)
- AllMiniLM models
- Nomic Embed models
- Multilingual E5 models
- MxbaiEmbed models

For a complete list, check the `/v1/params` endpoint.

## Documentation

API documentation is available at:
- Swagger UI: `/swagger`
- Redoc: `/redoc`
- RapiDoc: `/rapidoc`
- OpenAPI JSON: `/openapi.json`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see below for a summary:

GNU General Public License v3.0 (GPL-3.0)

Permissions:
- Commercial use
- Distribution
- Modification
- Patent use
- Private use

Conditions:
- Disclose source
- License and copyright notice
- Same license
- State changes

Limitations:
- Liability
- Warranty

For the full license text, see [LICENSE](LICENSE) or visit https://www.gnu.org/licenses/gpl-3.0.en.html

## 📧 Contact

support@sokratis.xyz
