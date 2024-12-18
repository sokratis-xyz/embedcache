use std::os::unix::process;
use std::sync::Arc;
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;
use async_sqlite::{JournalMode, PoolBuilder, Pool};
use sha2::{Sha256, Digest};
use anyhow::{Result, Context};
use fastembed::{TextEmbedding, EmbeddingModel, InitOptions};
use async_trait::async_trait;
use readability::extractor;
use tokio::task;
use std::collections::HashMap;
use apistos::{api_operation, ApiComponent};
use schemars::JsonSchema;
use actix_web::middleware::Logger;


use apistos::app::{BuildConfig, OpenApiWrapper};
use apistos::info::Info;
use apistos::server::Server;
use apistos::spec::Spec;
use apistos::web::{get, post, resource, scope};
use apistos::{RapidocConfig, RedocConfig, ScalarConfig, SwaggerUIConfig};

use async_sqlite::rusqlite::Error;
use dotenv::dotenv;
use std::env;

// Structures

#[derive(Debug, Serialize, Deserialize, Clone,JsonSchema, ApiComponent)]
struct Config {
    chunking_type: String,
    chunking_size: usize,
    embedding_model: String,
}

#[derive(Debug, Serialize, Deserialize,JsonSchema, ApiComponent)]
struct ProcessedContent {
    url: String,
    config: Config,
    chunks: HashMap<usize, String>,
    embeddings: HashMap<usize, Vec<f32>>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize,JsonSchema, ApiComponent)]
struct InputData {
    url: String,
    config: Option<Config>,
}

#[derive(Debug, Serialize, Deserialize,JsonSchema, ApiComponent)]
struct InputDataText {
    text: Vec<String>,
    config: Option<Config>,
}

struct AppState {
    db_pool: Pool,
    models: HashMap<String, Arc<TextEmbedding>>,
    chunkers: HashMap<String, Box<dyn ContentChunker + Send + Sync>>,
}

// Traits for extensibility

#[async_trait]
trait ContentChunker {
    async fn chunk(&self, content: &str, size: usize) -> Vec<String>;
}

#[async_trait]
trait Embedder {
    async fn embed(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>>;
}

// Chunker implementations

struct WordChunker;

#[async_trait]
impl ContentChunker for WordChunker {
    async fn chunk(&self, content: &str, size: usize) -> Vec<String> {
        content.split_whitespace()
            .collect::<Vec<&str>>()
            .chunks(size)
            .map(|chunk| chunk.join(" "))
            .collect()
    }
}

struct LLMConceptChunker;

#[async_trait]
impl ContentChunker for LLMConceptChunker {
    async fn chunk(&self, content: &str, size: usize) -> Vec<String> {
        // Placeholder implementation
        // TODO: Implement LLM-based concept chunking
        vec![content.to_string()]
    }
}

struct LLMIntrospectionChunker;

#[async_trait]
impl ContentChunker for LLMIntrospectionChunker {
    async fn chunk(&self, content: &str, size: usize) -> Vec<String> {
        // Placeholder implementation
        // TODO: Implement LLM-based introspection chunking
        vec![content.to_string()]
    }
}

struct FastEmbedder {
    model: Arc<TextEmbedding>,
}

#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let chunks = chunks.to_vec();

        task::spawn_blocking(move || {
            model.embed(chunks, None)
        })
        .await
        .map_err(|e| anyhow::Error::from(e))
        .and_then(|result| result.map_err(|e| anyhow::Error::from(e)))
    }
}

// Helper functions

fn get_default_config() -> Config {
    Config {
        chunking_type: "words".to_string(),
        chunking_size: 512,
        embedding_model: "BGESmallENV15".to_string(),
    }
}

fn generate_hash(url: &str, config: &Config) -> String {
    let mut hasher = Sha256::new();
    hasher.update(url);
    hasher.update(&config.chunking_type);
    hasher.update(config.chunking_size.to_string());
    hasher.update(&config.embedding_model);
    format!("{:x}", hasher.finalize())
}

// Main processing function
#[api_operation(summary = "Process a text and return the embeddings")]
async fn embed_text(
    input: web::Json<InputDataText>,
    data: web::Data<AppState>,
) -> Result<HttpResponse, actix_web::Error> {
    let config = input.config.clone().unwrap_or_else(get_default_config);
    let model = data.models.get(&config.embedding_model)
        .ok_or_else(|| actix_web::error::ErrorBadRequest(format!("Unsupported embedding model: {}", config.embedding_model)))?;
    let embedder = FastEmbedder { model: Arc::clone(model) };
    let embeddings = embedder.embed(&input.text).await.map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(embeddings))
}

// Main processing function
#[api_operation(summary = "Process a URL and return processed content")]
async fn process_url(
    input: web::Json<InputData>,
    data: web::Data<AppState>,
) -> Result<HttpResponse, actix_web::Error> {

    let config = input.config.clone().unwrap_or_else(get_default_config);
    let hash = generate_hash(&input.url, &config);

    // Check cache
    if let Some(cached_content) = get_from_cache(&data.db_pool, hash.clone()).await? {
        return Ok(HttpResponse::Ok().json(cached_content));
    }

    // Fetch content
    let content = fetch_content(input.url.clone()).await.map_err(actix_web::error::ErrorInternalServerError)?;

    if content == "Failed to scrape content" {
        let processed_content = ProcessedContent {
            url: input.url.clone(),
            config: config.clone(),
            chunks: HashMap::new(),
            embeddings: HashMap::new(),
            error: "Failed to scrape content".to_string().into(),
        };
        return Ok(HttpResponse::Ok().json(processed_content));
    }

    // Process content
    let chunker = data.chunkers.get(&config.chunking_type)
        .ok_or_else(|| actix_web::error::ErrorBadRequest(format!("Unsupported chunking type: {}", config.chunking_type)))?;
    let chunks = chunker.chunk(&content, config.chunking_size).await;

    let model = data.models.get(&config.embedding_model)
        .ok_or_else(|| actix_web::error::ErrorBadRequest(format!("Unsupported embedding model: {}", config.embedding_model)))?;
    let embedder = FastEmbedder { model: Arc::clone(model) };
    let embeddings = embedder.embed(&chunks).await.map_err(actix_web::error::ErrorInternalServerError)?;

    let processed_content = ProcessedContent {
        url: input.url.clone(),
        config: config.clone(),
        chunks: chunks.into_iter().enumerate().collect(),
        embeddings: embeddings.into_iter().enumerate().collect(),
        error: None,
    };

    // Cache result
    cache_result(&data.db_pool, hash.clone(), &processed_content).await?;

    Ok(HttpResponse::Ok().json(processed_content))
}

// Database functions

async fn get_from_cache(pool: &Pool, hash: String) -> Result<Option<ProcessedContent>, actix_web::Error> {
    let result: Option<String> = pool
        .conn(|conn| {
            conn.query_row("SELECT content FROM cache WHERE hash = ?", [hash], |row| row.get(0))
                .map(Some)
                .or_else(|err| match err {
                    Error::QueryReturnedNoRows => Ok(None),
                    _ => Err(err),
                })
        })
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(result.map(|json| serde_json::from_str(&json).unwrap()))
}


async fn cache_result(pool: &Pool, hash: String, content: &ProcessedContent) -> Result<(), actix_web::Error> {
    let json = serde_json::to_string(content).map_err(actix_web::error::ErrorInternalServerError)?;
    
    pool.conn(|conn| {
        conn.execute("INSERT OR REPLACE INTO cache (hash, content) VALUES (?, ?)", [hash, json])
    }).await.map_err(actix_web::error::ErrorInternalServerError)?;
    
    Ok(())
}

// Content fetching function using readability

async fn fetch_content(url: String) -> Result<String> {

    task::spawn_blocking(move || {

        extractor::scrape(&url)
            .map(|product| product.content)
            .unwrap_or_else(|_| String::from("Failed to scrape content"))
        
    }).await.context("Failed to fetch content")

}

// API to list supported models and chunking types
#[api_operation(summary = "Get a list of supported features")]
async fn list_supported_features() -> HttpResponse {
    let supported_features = json!({
        "chunking_types": ["words", "llm-concept", "llm-introspection"],
        "embedding_models": [
            "AllMiniLML6V2",
            "AllMiniLML6V2Q",
            "AllMiniLML12V2",
            "AllMiniLML12V2Q",
            "BGEBaseENV15",
            "BGEBaseENV15Q",
            "BGELargeENV15",
            "BGELargeENV15Q",
            "BGESmallENV15",
            "BGESmallENV15Q",
            "NomicEmbedTextV1",
            "NomicEmbedTextV15",
            "NomicEmbedTextV15Q",
            "ParaphraseMLMiniLML12V2",
            "ParaphraseMLMiniLML12V2Q",
            "ParaphraseMLMpnetBaseV2",
            "BGESmallZHV15",
            "MultilingualE5Small",
            "MultilingualE5Base",
            "MultilingualE5Large",
            "MxbaiEmbedLargeV1",
            "MxbaiEmbedLargeV1Q"
        ]
    });

    HttpResponse::Ok().json(supported_features)
}

// Main function

fn get_embedding_model(model_name: &str) -> Option<EmbeddingModel> {
    match model_name {
        "AllMiniLML6V2" => Some(EmbeddingModel::AllMiniLML6V2),
        "AllMiniLML6V2Q" => Some(EmbeddingModel::AllMiniLML6V2Q),
        "AllMiniLML12V2" => Some(EmbeddingModel::AllMiniLML12V2),
        "AllMiniLML12V2Q" => Some(EmbeddingModel::AllMiniLML12V2Q),
        "BGEBaseENV15" => Some(EmbeddingModel::BGEBaseENV15),
        "BGEBaseENV15Q" => Some(EmbeddingModel::BGEBaseENV15Q),
        "BGELargeENV15" => Some(EmbeddingModel::BGELargeENV15),
        "BGELargeENV15Q" => Some(EmbeddingModel::BGELargeENV15Q),
        "BGESmallENV15" => Some(EmbeddingModel::BGESmallENV15),
        "BGESmallENV15Q" => Some(EmbeddingModel::BGESmallENV15Q),
        "NomicEmbedTextV1" => Some(EmbeddingModel::NomicEmbedTextV1), 
        "NomicEmbedTextV15" => Some(EmbeddingModel::NomicEmbedTextV15),
        "NomicEmbedTextV15Q" => Some(EmbeddingModel::NomicEmbedTextV15Q),
        "ParaphraseMLMiniLML12V2" => Some(EmbeddingModel::ParaphraseMLMiniLML12V2),
        "ParaphraseMLMiniLML12V2Q" => Some(EmbeddingModel::ParaphraseMLMiniLML12V2Q),
        "ParaphraseMLMpnetBaseV2" => Some(EmbeddingModel::ParaphraseMLMpnetBaseV2),
        "BGESmallZHV15" => Some(EmbeddingModel::BGESmallZHV15),
        "MultilingualE5Small" => Some(EmbeddingModel::MultilingualE5Small),
        "MultilingualE5Base" => Some(EmbeddingModel::MultilingualE5Base),
        "MultilingualE5Large" => Some(EmbeddingModel::MultilingualE5Large),
        "MxbaiEmbedLargeV1" => Some(EmbeddingModel::MxbaiEmbedLargeV1),
        "MxbaiEmbedLargeV1Q" => Some(EmbeddingModel::MxbaiEmbedLargeV1Q),
        _ => None,
    }
}

struct ServerConfig {
    host: String,
    port: u16,
    db_path: String,
    db_journal_mode: String,
    enabled_models: Vec<String>,
}

impl ServerConfig {
    fn from_env() -> Result<Self, std::env::VarError> {
        Ok(Self {
            host: env::var("SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: env::var("SERVER_PORT").unwrap_or_else(|_| "8081".to_string())
                .parse()
                .expect("Invalid SERVER_PORT"),
            db_path: env::var("DB_PATH").unwrap_or_else(|_| "cache.db".to_string()),
            db_journal_mode: env::var("DB_JOURNAL_MODE").unwrap_or_else(|_| "wal".to_string()),
            enabled_models: env::var("ENABLED_MODELS")
                .unwrap_or_else(|_| "AllMiniLML6V2".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
        })
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Load configuration
    let config = ServerConfig::from_env().expect("Failed to load configuration");

    let db_pool = PoolBuilder::new()
        .path(&config.db_path)
        .journal_mode(match config.db_journal_mode.to_lowercase().as_str() {
            "wal" => JournalMode::Wal,
            "truncate" => JournalMode::Truncate,
            "persist" => JournalMode::Persist,
            _ => JournalMode::Wal,
        })
        .open()
        .await
        .expect("Failed to create database pool");
    
    // Initialize database
    db_pool.conn(|conn| {
        conn.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, content TEXT)", [])
    }).await.expect("Failed to create cache table");

    // Initialize models
    let mut models = HashMap::new();
    
    for name in &config.enabled_models {
        let model = Arc::new(TextEmbedding::try_new(InitOptions {
            model_name: get_embedding_model(name).expect("Invalid model name"),
            show_download_progress: true,
            ..Default::default()
        }).expect("Failed to create TextEmbedding model"));
        models.insert(name.to_string(), model);
    }

    // Initialize chunkers
    let mut chunkers: HashMap<String, Box<dyn ContentChunker + Send + Sync>> = HashMap::new();
    chunkers.insert("words".to_string(), Box::new(WordChunker));
    chunkers.insert("llm-concept".to_string(), Box::new(LLMConceptChunker));
    chunkers.insert("llm-introspection".to_string(), Box::new(LLMIntrospectionChunker));

    let app_state = web::Data::new(AppState { db_pool, models, chunkers });

    let server_addr = format!("{}:{}", config.host, config.port);
    println!("Starting server at {}", server_addr);

    HttpServer::new(move || {
        let spec = Spec {
            info: Info {
                title: "Embedcache API".to_string(),
                description: Some("This is the embed cache API!".to_string()),
                ..Default::default()
            },
            servers: vec![Server {
                url: "/".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        App::new()
            .document(spec)
            .wrap(Logger::default())
            .app_data(app_state.clone())
            .service(scope("/v1")
                .service(resource("/embed").route(post().to(embed_text)))
                .service(resource("/process").route(post().to(process_url)))
                .service(resource("/params").route(get().to(list_supported_features)))
            )
            .build_with(
                "/openapi.json",
                BuildConfig::default()
                    .with(RapidocConfig::new(&"/rapidoc"))
                    .with(RedocConfig::new(&"/redoc"))
                    .with(ScalarConfig::new(&"/scalar"))
                    .with(SwaggerUIConfig::new(&"/swagger")),
            )
    })
    .bind(server_addr)?
    .run()
    .await
}