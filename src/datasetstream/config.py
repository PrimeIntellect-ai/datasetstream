from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Any, Final

from datasetstream.dataset import DatasetConfig, TokenizerConfig

# Default values for server configuration
DEFAULT_HOST: Final = "localhost"
DEFAULT_PORT: Final = 8765
DEFAULT_TOKEN_SIZE_BITS: Final = 16
DEFAULT_DOC_SEPARATOR: Final = 50256
DEFAULT_VOCAB_SIZE: Final = 50257

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

@dataclass(frozen=True)
class ServerConfig:
    """Server configuration including dataset mappings"""
    host: str
    port: int
    datasets_dir: Path
    dataset_configs: Dict[str, DatasetConfig]
    
    @classmethod
    def from_json(cls, config_path: Path) -> 'ServerConfig':
        """Create server configuration from a JSON file
        
        :param config_path: Path to JSON configuration file
        :return: Validated ServerConfig instance
        :raises:
            ConfigError: If configuration is invalid or missing required fields
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file isn't valid JSON
        """
        try:
            with open(config_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in configuration file: {e}")
            
        try:
            datasets = {}
            datasets_dir = Path(data.get("datasets_dir", "data"))
            
            for dataset_id, dataset_data in data.get("datasets", {}).items():
                if '/' in dataset_id:
                    raise ConfigError(f"Invalid dataset ID: {dataset_id}; must not contain '/'")
                if not isinstance(dataset_data, dict):
                    raise ConfigError(f"Invalid dataset configuration for '{dataset_id}'")
                    
                tokenizer_data = dataset_data.get("tokenizer", {})
                if not isinstance(tokenizer_data, dict):
                    raise ConfigError(f"Invalid tokenizer configuration for dataset '{dataset_id}'")
                
                data_files = dataset_data.get("data_files")
                if not data_files:
                    raise ConfigError(f"Missing data_files for dataset '{dataset_id}'")
                
                tokenizer_config = TokenizerConfig(
                    document_separator_token=tokenizer_data.get("document_separator_token", DEFAULT_DOC_SEPARATOR),
                    vocab_size=tokenizer_data.get("vocab_size", DEFAULT_VOCAB_SIZE)
                )

                datasets[dataset_id] = DatasetConfig(
                    data_files=[datasets_dir / data_file for data_file in data_files],
                    tokenizer_config=tokenizer_config,
                    token_size_bits=dataset_data.get("token_size_bits", DEFAULT_TOKEN_SIZE_BITS)
                )
            
            config = ServerConfig(
                host=data.get("host", DEFAULT_HOST),
                port=data.get("port", DEFAULT_PORT),
                datasets_dir=datasets_dir,
                dataset_configs=datasets
            )
            
            config.validate()
            return config
            
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid configuration: {e}")
    
    def validate(self) -> None:
        """Validate the server configuration
        
        :raises ConfigError: If any validation checks fail
        """
        if not self.datasets_dir.exists():
            raise ConfigError(f"Datasets directory does not exist: {self.datasets_dir}")

        if not self.dataset_configs:
            raise ConfigError("No datasets configured")

        for dataset_id, dataset_config in self.dataset_configs.items():
            if any([not path.exists() for path in dataset_config.data_files]):
                raise ConfigError(
                    f"Data file for dataset '{dataset_id}' does not exist: "
                    f"{dataset_config.data_files}"
                )

@dataclass
class ServerState:
    """Runtime state tracking for the dataset server"""
    config: ServerConfig
    active_connections: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization"""
        if not isinstance(self.config, ServerConfig):
            raise TypeError("config must be a ServerConfig instance")
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific dataset
        
        :param dataset_id: Identifier of the dataset
        :return: Dict containing dataset metadata or None if dataset not found
        """
        if dataset_id not in self.config.dataset_configs:
            return None
            
        dataset_config = self.config.dataset_configs[dataset_id]
        return {
            "id": dataset_id,
            "file_paths": [str(path) for path in dataset_config.data_files],
            "file_sizes": [path.stat().st_size for path in dataset_config.data_files],
            "active_connections": self.active_connections.get(dataset_id, 0),
            "tokenizer": {
                "token_size_bits": dataset_config.token_size_bits,
                "vocab_size": dataset_config.tokenizer_config.vocab_size,
                "document_separator_token": dataset_config.tokenizer_config.document_separator_token
            }
        }
    
    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about all available datasets
        
        :return: Dict mapping dataset IDs to their metadata
        """
        return {
            dataset_id: self.get_dataset_info(dataset_id)
            for dataset_id in self.config.dataset_configs
        }
    
    def increment_connections(self, dataset_id: str) -> bool:
        """Increment connection count for a dataset
        
        :param dataset_id: Identifier of the dataset
        :return: True if successful, False if dataset not found
        """
        if dataset_id not in self.config.dataset_configs:
            return False
        self.active_connections[dataset_id] = self.active_connections.get(dataset_id, 0) + 1
        return True
    
    def decrement_connections(self, dataset_id: str) -> bool:
        """Decrement connection count for a dataset
        
        :param dataset_id: Identifier of the dataset
        :return: True if successful, False if dataset not found
        """
        if dataset_id not in self.config.dataset_configs:
            return False
        current = self.active_connections.get(dataset_id, 0)
        if current > 0:
            self.active_connections[dataset_id] = current - 1
        return True 